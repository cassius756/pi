/**
 * Agent Vitals HUD - live telemetry footer for pi.
 *
 * Replaces the footer with four always-visible segments:
 *
 *   - Context battery: fine-grained eighth-block bar that shifts
 *     success -> warning -> error as the context window fills, with a flag
 *     once usage enters the auto-compact danger zone (>= 80%).
 *   - Throughput sparkline: truecolor gradient bars over the last six
 *     seconds of streamed output plus a tokens/second estimate. The
 *     char->token ratio self-calibrates against each finished message's
 *     reported usage, so the estimate converges per provider tokenizer.
 *   - Cache hit rate: share of input tokens served from the prompt cache.
 *   - Session cost.
 *
 * Usage:
 *   pi --extension examples/extensions/agent-vitals.ts
 *
 * Commands:
 *   /vitals            toggle the HUD
 *   /vitals on|off     set explicitly
 */

import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

/** A streamed-character sample; the unit the sparkline and tps estimate are built from. */
interface Sample {
	t: number;
	chars: number;
}

const BUCKETS = 12;
const BUCKET_MS = 500;
const WINDOW_MS = BUCKETS * BUCKET_MS;
/** Gauge cells; eighth blocks give 12 * 8 = 1/96 window resolution. */
const GAUGE_CELLS = 12;
const EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
const SPARK = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
/** Sparkline gradient endpoints, teal -> violet, as 24-bit RGB. */
const GRADIENT_FROM = { r: 20, g: 184, b: 166 };
const GRADIENT_TO = { r: 139, g: 92, b: 246 };

function clamp(n: number, lo: number, hi: number): number {
	return Math.min(hi, Math.max(lo, n));
}

function fmtTokens(n: number): string {
	return n < 1000 ? `${n}` : `${(n / 1000).toFixed(n < 100000 ? 1 : 0)}k`;
}

function gradient(i: number, n: number, text: string): string {
	const t = n <= 1 ? 0 : i / (n - 1);
	const r = Math.round(GRADIENT_FROM.r + (GRADIENT_TO.r - GRADIENT_FROM.r) * t);
	const g = Math.round(GRADIENT_FROM.g + (GRADIENT_TO.g - GRADIENT_FROM.g) * t);
	const b = Math.round(GRADIENT_FROM.b + (GRADIENT_TO.b - GRADIENT_FROM.b) * t);
	return `\x1b[38;2;${r};${g};${b}m${text}\x1b[39m`;
}

export default function (pi: ExtensionAPI) {
	let enabled = false;
	let ctxRef: ExtensionContext | undefined;
	/** Streamed-character samples inside the rolling window. */
	let samples: Sample[] = [];
	/** Chars streamed by the assistant message currently in flight. */
	let messageChars = 0;
	/** Estimated tokens per streamed char; refined on every finished message. */
	let tokensPerChar = 0.25;
	let tps = 0;
	/** Session-max chars per bucket; decays so an old burst does not flatten the bars. */
	let sparkMax = 80;

	function install(ctx: ExtensionContext) {
		ctx.ui.setFooter((_tui, theme) => ({
			dispose() {
				samples = [];
			},
			invalidate() {},
			render(width: number): string[] {
				const c = ctxRef;
				if (!c) return [""];

				const now = Date.now();
				while (samples.length > 0 && now - samples[0].t > WINDOW_MS) samples.shift();

				// Segments in drop order: cost drops first, gauge last.
				const segments: { text: string; rank: number }[] = [];

				// Context battery.
				const usage = c.getContextUsage();
				if (usage && usage.percent != null) {
					const percent = clamp(usage.percent, 0, 100);
					const color = percent < 60 ? "success" : percent < 80 ? "warning" : "error";
					const fill = (percent / 100) * GAUGE_CELLS;
					const full = Math.floor(fill);
					const frac = Math.round((fill - full) * 8);
					let gauge = "";
					for (let i = 0; i < GAUGE_CELLS; i++) {
						if (i < full) gauge += theme.fg(color, "█");
						else if (i === full && frac > 0) gauge += theme.fg(color, EIGHTHS[frac]);
						else gauge += theme.fg("dim", "░");
					}
					const flag = percent >= 80 ? theme.fg("warning", " ⚑") : "";
					const left = usage.contextWindow - (usage.tokens ?? 0);
					segments.push({
						text: `${gauge} ${theme.fg(color, `${Math.round(percent)}%`)}${flag} ${theme.fg("dim", `${fmtTokens(Math.max(0, left))} free`)}`,
						rank: 5,
					});
				} else {
					segments.push({
						text: theme.fg("dim", `${"░".repeat(GAUGE_CELLS)} ctx n/a`),
						rank: 5,
					});
				}

				// Throughput sparkline + self-calibrating tps.
				if (samples.length > 0) {
					const buckets = new Array<number>(BUCKETS).fill(0);
					for (const s of samples) {
						const idx = BUCKETS - 1 - Math.floor((now - s.t) / BUCKET_MS);
						if (idx >= 0 && idx < BUCKETS) buckets[idx] += s.chars;
					}
					sparkMax = Math.max(40, Math.max(...buckets), sparkMax * 0.97);
					let bars = "";
					for (let i = 0; i < BUCKETS; i++) {
						const level = clamp(Math.round((buckets[i] / sparkMax) * (SPARK.length - 1)), 0, SPARK.length - 1);
						bars += gradient(i, BUCKETS, level === 0 ? "·" : SPARK[level]);
					}
					const recent = samples.reduce((sum, s) => sum + (now - s.t < 1000 ? s.chars : 0), 0);
					tps = tps * 0.6 + recent * tokensPerChar * 0.4;
					segments.push({ text: `${bars} ${theme.fg("muted", `${tps.toFixed(0)}t/s`)}`, rank: 4 });
				}

				// Cache hit rate and session cost from the transcript.
				let input = 0;
				let cacheRead = 0;
				let cost = 0;
				for (const e of c.sessionManager.getBranch()) {
					if (e.type === "message" && e.message.role === "assistant") {
						const m = e.message as AssistantMessage;
						input += m.usage.input;
						cacheRead += m.usage.cacheRead;
						cost += m.usage.cost.total;
					}
				}
				const denom = cacheRead + input;
				if (denom > 0) {
					segments.push({
						text: `${theme.fg("accent", "cache")} ${theme.fg("muted", `${Math.round((cacheRead / denom) * 100)}%`)}`,
						rank: 2,
					});
				}
				segments.push({ text: theme.fg("dim", `$${cost.toFixed(cost < 10 ? 4 : 2)}`), rank: 1 });

				// Fit to width by dropping the least important segments first.
				const sep = theme.fg("dim", " │ ");
				let kept = [...segments];
				let line = kept.map((s) => s.text).join(sep);
				for (;;) {
					const widthOk = visibleWidth(line) <= width;
					if (widthOk) break;
					const drop = kept.filter((s) => s.rank < 5).sort((a, b) => a.rank - b.rank)[0];
					if (!drop) break;
					kept = kept.filter((s) => s !== drop);
					line = kept.map((s) => s.text).join(sep);
				}
				return [truncateToWidth(line, width)];
			},
		}));
	}

	function uninstall(ctx: ExtensionContext) {
		ctx.ui.setFooter(undefined);
	}

	pi.on("session_start", async (_event, ctx) => {
		ctxRef = ctx;
		if (enabled) install(ctx);
	});

	pi.on("message_start", async (event) => {
		if (event.message.role === "assistant") messageChars = 0;
	});

	pi.on("message_update", async (event) => {
		const e = event.assistantMessageEvent;
		if (e.type !== "text_delta" && e.type !== "thinking_delta" && e.type !== "toolcall_delta") return;
		const chars = e.delta.length;
		messageChars += chars;
		const now = Date.now();
		const last = samples[samples.length - 1];
		if (last && now - last.t < 50) last.chars += chars;
		else samples.push({ t: now, chars });
	});

	pi.on("message_end", async (event) => {
		if (event.message.role !== "assistant") return;
		const usage = (event.message as AssistantMessage).usage;
		if (messageChars > 1000 && usage.output > 0) {
			const observed = usage.output / messageChars;
			// EMA so one provider's tokenizer quirks do not jerk the estimate around.
			tokensPerChar = clamp(tokensPerChar * 0.7 + observed * 0.3, 0.1, 0.6);
		}
		messageChars = 0;
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		samples = [];
		if (enabled) uninstall(ctx);
	});

	pi.registerCommand("vitals", {
		description: "Toggle the agent vitals HUD footer (context, throughput, cache, cost)",
		handler: async (args, ctx) => {
			const arg = args.trim().toLowerCase();
			if (arg === "on") enabled = true;
			else if (arg === "off") enabled = false;
			else enabled = !enabled;

			if (enabled) {
				install(ctx);
				ctx.ui.notify("Vitals HUD enabled", "info");
			} else {
				uninstall(ctx);
				ctx.ui.notify("Vitals HUD disabled", "info");
			}
		},
	});
}
