/**
 * Context Map - a disk-usage analyzer for the model context window.
 *
 * `/contextmap` opens an overlay that breaks the active context down by
 * category (system prompt, user messages, replies + reasoning, and per-tool
 * outputs) as a stacked bar plus a legend. Tool outputs are usually the
 * biggest and least visible context consumers; this makes them visible and
 * surgically trimmable:
 *
 *   - Enter expands a category into its largest entries.
 *   - `t` marks a tool-result entry for trimming. Trims are submitted as
 *     `context_edit` entries at the next `turn_end` boundary, which removes
 *     the entry's content from model context while the transcript stays
 *     intact (the session is append-only).
 *
 * Per-message estimates come from pi-ai's char-based token estimator and are
 * calibrated against the provider-reported context usage, so the map's
 * proportions track the real window.
 *
 * Usage:
 *   pi --extension examples/extensions/context-map.ts
 *
 * Commands:
 *   /contextmap    open the map
 */

import type {
	AssistantMessage,
	ImageContent,
	Message,
	TextContent,
	ToolResultMessage,
	UserMessage,
} from "@earendil-works/pi-ai";
import { estimateMessageTokens } from "@earendil-works/pi-ai/utils/estimate";
import type {
	ContextEditEntryDraft,
	ContextUsage,
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
	Theme,
} from "@earendil-works/pi-coding-agent";
import type { Focusable } from "@earendil-works/pi-tui";
import { matchesKey, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

interface MapEntry {
	id: string;
	label: string;
	tokens: number;
	toolName?: string;
}

interface Category {
	key: string;
	label: string;
	color: string;
	tokens: number;
	entries: MapEntry[];
	expandable: boolean;
}

interface ContextMap {
	categories: Category[];
	/** Estimated context tokens after pending trims. */
	total: number;
	/** Multiplier aligning char-based estimates with provider-reported usage; 1 when unavailable. */
	scale: number;
	/** Active model context window, when known. */
	contextWindow: number | undefined;
}

/** Truecolor category palette; chrome uses theme tokens. */
const COLORS: Record<string, string> = {
	system: "#94a3b8",
	user: "#22d3ee",
	assistant: "#34d399",
	read: "#7dd3fc",
	bash: "#fbbf24",
	edit: "#a5b4fc",
	write: "#f472b6",
	other: "#fb923c",
};
/** Palette cycled through for dominant non-built-in tools. */
const PROMOTED_COLORS = ["#fca5a5", "#d8b4fe", "#86efac", "#fde047", "#5eead4"];
const KNOWN_TOOL_CATEGORIES = new Set(["read", "bash", "edit", "write"]);
/** Non-built-in tools above this share of the window get their own category. */
const PROMOTE_SHARE = 0.08;

function fg(hex: string, text: string): string {
	const r = Number.parseInt(hex.slice(1, 3), 16);
	const g = Number.parseInt(hex.slice(3, 5), 16);
	const b = Number.parseInt(hex.slice(5, 7), 16);
	return `\x1b[38;2;${r};${g};${b}m${text}\x1b[39m`;
}

function fmtK(n: number): string {
	return n < 1000 ? `${Math.round(n)}` : `${(n / 1000).toFixed(1)}k`;
}

function jsonChars(value: unknown): number {
	try {
		return JSON.stringify(value)?.length ?? 0;
	} catch {
		return 0;
	}
}

/** Char-based token estimate that also tolerates custom agent message roles. */
function messageTokens(m: Message): number {
	try {
		return estimateMessageTokens(m);
	} catch {
		// Custom roles can carry blocks the standard estimator cannot iterate
		// (e.g. no `name` field) or non-iterable content; fall back to JSON size.
		return Math.ceil(jsonChars(m) / 4);
	}
}

/** First printable line of a message's content, for entry labels. */
function firstLine(content: string | (TextContent | ImageContent)[]): string {
	if (typeof content === "string") {
		const line = content.split("\n").find((l) => l.trim().length > 0);
		return (line ?? "").trim();
	}
	for (const block of content) {
		if (block.type !== "text") continue;
		const line = block.text.split("\n").find((l) => l.trim().length > 0);
		if (line) return line.trim();
	}
	return "[image]";
}

function buildMap(
	sessionManager: ExtensionContext["sessionManager"],
	usage: ContextUsage | undefined,
	pending: ReadonlySet<string>,
): ContextMap {
	const categories = new Map<string, Category>();
	const cat = (key: string, label: string, expandable: boolean, color?: string): Category => {
		let c = categories.get(key);
		if (!c) {
			c = { key, label, color: color ?? COLORS[key] ?? COLORS.other, tokens: 0, entries: [], expandable };
			categories.set(key, c);
		}
		return c;
	};

	// buildContextEntries is compaction-aware; honor earlier null-replacement edits too.
	const contextEntries = sessionManager.buildContextEntries();
	const editedOut = new Set<string>();
	for (const e of contextEntries) {
		if (e.type === "context_edit" && e.replacement === null) editedOut.add(e.targetId);
	}

	for (const e of contextEntries) {
		if (e.type !== "message" || editedOut.has(e.id)) continue;
		const m = e.message as Message;
		const tokens = messageTokens(m);
		if (m.role === "system") {
			cat("system", "system + tools", false).tokens += tokens;
		} else if (m.role === "user") {
			cat("user", "user", false).entries.push({
				id: e.id,
				label: firstLine((m as UserMessage).content),
				tokens,
			});
		} else if (m.role === "assistant") {
			const label = firstLine((m as AssistantMessage).content.filter((b): b is TextContent => b.type === "text"));
			cat("assistant", "replies + reasoning", false).entries.push({ id: e.id, label, tokens });
		} else if (m.role === "toolResult") {
			const tr = m as ToolResultMessage;
			const key = KNOWN_TOOL_CATEGORIES.has(tr.toolName) ? tr.toolName : "other";
			cat(key, key === "other" ? "other tools" : key, true).entries.push({
				id: e.id,
				label: firstLine(tr.content),
				tokens,
				toolName: tr.toolName,
			});
		}
	}

	for (const c of categories.values()) {
		c.tokens = c.entries.reduce((sum, entry) => sum + entry.tokens, 0);
	}

	// Promote dominant non-built-in tools out of the "other" bucket.
	const other = categories.get("other");
	if (other && other.tokens > 0) {
		const total0 = [...categories.values()].reduce((sum, c) => sum + c.tokens, 0);
		const byTool = new Map<string, MapEntry[]>();
		for (const entry of other.entries) {
			const name = entry.toolName ?? "?";
			const list = byTool.get(name) ?? [];
			list.push(entry);
			byTool.set(name, list);
		}
		let promoted = 0;
		for (const [name, entries] of byTool) {
			const tokens = entries.reduce((sum, entry) => sum + entry.tokens, 0);
			if (tokens / Math.max(1, total0) < PROMOTE_SHARE) continue;
			const c = cat(`tool:${name}`, name, true, PROMOTED_COLORS[promoted % PROMOTED_COLORS.length]);
			c.entries = entries;
			c.tokens = tokens;
			promoted++;
		}
		if (promoted > 0) {
			const promotedIds = new Set(
				[...categories.values()].filter((c) => c.key.startsWith("tool:")).flatMap((c) => c.entries),
			);
			other.entries = other.entries.filter((entry) => !promotedIds.has(entry));
			other.tokens = other.entries.reduce((sum, entry) => sum + entry.tokens, 0);
			if (other.entries.length === 0) categories.delete("other");
		}
	}

	// Calibrate char-based estimates against provider-reported usage, then apply
	// pending trims so every number the map shows is the post-trim projection.
	const estTotal = [...categories.values()].reduce((sum, c) => sum + c.tokens, 0) || 1;
	const reported = usage?.tokens;
	const scale = reported && reported > 0 ? Math.min(4, Math.max(0.25, reported / estTotal)) : 1;

	for (const c of categories.values()) {
		c.tokens *= scale;
		for (const entry of c.entries) entry.tokens *= scale;
		c.entries.sort((a, b) => b.tokens - a.tokens);
		c.tokens -= c.entries.filter((entry) => pending.has(entry.id)).reduce((sum, entry) => sum + entry.tokens, 0);
	}

	return {
		categories: [...categories.values()].filter((c) => c.tokens > 0).sort((a, b) => b.tokens - a.tokens),
		total: [...categories.values()].reduce((sum, c) => sum + c.tokens, 0),
		scale,
		contextWindow: usage?.contextWindow,
	};
}

type Row = { kind: "category"; cat: Category } | { kind: "entry"; entry: MapEntry; trimmable: boolean };

class ContextMapComponent implements Focusable {
	readonly width = 78;
	focused = false;

	private selected = 0;
	private expanded = new Set<string>();
	private rows: Row[] = [];
	private theme: Theme;
	private map: ContextMap;
	private pending: Set<string>;
	private refreshMap: () => ContextMap;
	private done: () => void;

	constructor(theme: Theme, refreshMap: () => ContextMap, pending: Set<string>, done: () => void) {
		this.theme = theme;
		this.refreshMap = refreshMap;
		this.map = refreshMap();
		this.pending = pending;
		this.done = done;
		this.rebuildRows();
	}

	/** Stable identity of the selected row, so refreshes that re-sort keep the cursor in place. */
	private selectionKey(): string | undefined {
		const row = this.rows[this.selected];
		if (!row) return undefined;
		return row.kind === "category" ? `cat:${row.cat.key}` : `entry:${row.entry.id}`;
	}

	private rebuildRows(): void {
		const focus = this.selectionKey();
		this.rows = [];
		for (const c of this.map.categories) {
			this.rows.push({ kind: "category", cat: c });
			if (this.expanded.has(c.key)) {
				for (const entry of c.entries.slice(0, 8)) {
					this.rows.push({ kind: "entry", entry, trimmable: entry.toolName !== undefined });
				}
			}
		}
		this.selected = 0;
		if (focus) {
			const idx = this.rows.findIndex((r) =>
				r.kind === "category" ? `cat:${r.cat.key}` === focus : `entry:${r.entry.id}` === focus,
			);
			if (idx >= 0) this.selected = idx;
		}
		this.selected = Math.min(this.selected, Math.max(0, this.rows.length - 1));
	}

	handleInput(data: string): void {
		if (matchesKey(data, "escape")) {
			this.done();
			return;
		}
		if (matchesKey(data, "up")) {
			this.selected = Math.max(0, this.selected - 1);
			return;
		}
		if (matchesKey(data, "down")) {
			this.selected = Math.min(this.rows.length - 1, this.selected + 1);
			return;
		}
		if (matchesKey(data, "return")) {
			const row = this.rows[this.selected];
			if (row?.kind === "category" && row.cat.expandable) {
				if (this.expanded.has(row.cat.key)) this.expanded.delete(row.cat.key);
				else this.expanded.add(row.cat.key);
				this.rebuildRows();
			}
			return;
		}
		if (data === "t") {
			const row = this.rows[this.selected];
			if (row?.kind === "entry" && row.trimmable) {
				if (this.pending.has(row.entry.id)) this.pending.delete(row.entry.id);
				else this.pending.add(row.entry.id);
				// Rebuild so bar, totals, and legend stay the live post-trim projection.
				this.map = this.refreshMap();
				this.rebuildRows();
			}
		}
	}

	render(_width: number): string[] {
		const th = this.theme;
		const innerW = this.width - 2;
		const pad = (s: string) => s + " ".repeat(Math.max(0, innerW - visibleWidth(s)));
		const row = (s: string) => th.fg("border", "│") + truncateToWidth(pad(s), innerW) + th.fg("border", "│");
		const lines: string[] = [];
		const total = Math.max(1, this.map.total);

		const fill = this.map.contextWindow ? `${((this.map.total / this.map.contextWindow) * 100).toFixed(0)}%` : "?";
		const trimNote = this.pending.size > 0 ? th.fg("warning", ` ⌫${this.pending.size} pending`) : "";
		const calib = this.map.scale === 1 ? th.fg("dim", " est") : "";
		lines.push(th.fg("border", `╭${"─".repeat(innerW)}╮`));
		lines.push(
			row(
				` ${th.fg("accent", "Context Map")}  ${fmtK(this.map.total)}${calib} / ${this.map.contextWindow ? fmtK(this.map.contextWindow) : "?"} tokens (${fill})${trimNote}`,
			),
		);
		lines.push(row(""));

		// Stacked bar: one colored run per category, newest-largest first.
		const barW = innerW - 4;
		let bar = "";
		let used = 0;
		for (const c of this.map.categories) {
			const w = Math.min(Math.round((c.tokens / total) * barW), barW - used);
			if (w <= 0) continue;
			bar += fg(c.color, "█".repeat(w));
			used += w;
		}
		bar += th.fg("dim", "░".repeat(Math.max(0, barW - used)));
		lines.push(row(`  ${bar}`));
		lines.push(row(""));

		for (let i = 0; i < this.rows.length; i++) {
			const r = this.rows[i];
			if (!r) continue;
			const isSelected = i === this.selected;
			const cursor = isSelected ? "▶ " : "  ";

			if (r.kind === "category") {
				const share = ((r.cat.tokens / total) * 100).toFixed(0);
				const expandHint = r.cat.expandable ? (this.expanded.has(r.cat.key) ? "▾" : "▸") : " ";
				const miniBar = fg(r.cat.color, "█".repeat(Math.max(1, Math.round((r.cat.tokens / total) * 20))));
				lines.push(
					row(
						` ${cursor}${fg(r.cat.color, expandHint)} ${r.cat.label.padEnd(18, " ")}${fmtK(r.cat.tokens).padStart(7)}  ${share.padStart(3)}% ${miniBar}`,
					),
				);
			} else {
				const trimmed = this.pending.has(r.entry.id);
				const marker = trimmed ? th.fg("warning", "⌫") : r.trimmable ? th.fg("dim", "·") : " ";
				const share = ((r.entry.tokens / total) * 100).toFixed(0);
				const label = trimmed ? th.fg("dim", r.entry.label || "(empty)") : r.entry.label || "(empty)";
				const indent = isSelected ? "▶     " : "      ";
				lines.push(
					row(
						` ${indent}${marker} ${truncateToWidth(label, 36)} ${th.fg("dim", `${fmtK(r.entry.tokens).padStart(7)}  ${share.padStart(3)}%`)}`,
					),
				);
			}
		}

		lines.push(row(""));
		const note =
			this.pending.size > 0
				? th.fg("warning", `${this.pending.size} trim(s) apply at the end of the next turn`)
				: th.fg("dim", "trims submit as context_edit entries at the next turn boundary");
		lines.push(row(` ${note}`));
		lines.push(row(` ${th.fg("dim", "↑↓ navigate • ⏎ expand • t trim tool output • Esc close")}`));
		lines.push(th.fg("border", `╰${"─".repeat(innerW)}╯`));
		return lines;
	}

	invalidate(): void {}
	dispose(): void {}
}

export default function (pi: ExtensionAPI) {
	/** Entry ids marked for trimming, submitted at the next turn boundary. */
	const pending = new Set<string>();

	pi.on("session_start", async () => {
		pending.clear();
	});

	pi.on("session_shutdown", async () => {
		pending.clear();
	});

	// Turn boundaries are the only place session edits can be proposed; submits
	// queued trims as append-only context_edit entries without a follow-up call.
	pi.on("turn_end", async (_event, ctx) => {
		if (pending.size === 0) return undefined;
		const onBranch = new Set(ctx.sessionManager.getBranch().map((e) => e.id));
		const entries: ContextEditEntryDraft[] = [...pending]
			.filter((id) => onBranch.has(id))
			.map((targetId) => ({ type: "context_edit", targetId, replacement: null }));
		pending.clear();
		if (entries.length === 0) return undefined;
		return { entries };
	});

	pi.registerCommand("contextmap", {
		description: "Show where the context window is spent; trim tool outputs",
		handler: async (_args: string, ctx: ExtensionCommandContext) => {
			if (ctx.mode !== "tui") {
				ctx.ui.notify("context map requires interactive mode", "error");
				return;
			}
			const before = pending.size;
			const refreshMap = () => buildMap(ctx.sessionManager, ctx.getContextUsage(), pending);
			await ctx.ui.custom<void>((_tui, theme) => new ContextMapComponent(theme, refreshMap, pending, () => {}), {
				overlay: true,
			});
			if (pending.size !== before && pending.size > 0) {
				ctx.ui.notify(`${pending.size} trim(s) pending; applied at the end of the next turn`, "info");
			}
		},
	});
}
