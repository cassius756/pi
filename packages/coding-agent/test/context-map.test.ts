import type { AssistantMessage, Message, SystemMessage, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import { estimateMessageTokens } from "@earendil-works/pi-ai/utils/estimate";
import { describe, expect, it } from "vitest";
import { buildMap, ContextMapComponent, default as contextMapExtension } from "../examples/extensions/context-map.ts";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "../src/core/extensions/types.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import type { Theme } from "../src/modes/interactive/theme/theme.ts";

function system(text: string): SystemMessage {
	return { role: "system", content: text, timestamp: Date.now() };
}

function user(text: string): UserMessage {
	return { role: "user", content: text, timestamp: Date.now() };
}

function assistant(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "faux",
		provider: "faux",
		model: "faux",
		usage: {
			input: 10,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 11,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function toolResult(toolName: string, text: string): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: `call-${toolName}-${Math.random().toString(36).slice(2, 8)}`,
		toolName,
		content: [{ type: "text", text }],
		isError: false,
		timestamp: Date.now(),
	};
}

/** Readable first line plus bulk filler, so entry labels stay short while token counts stay large. */
function bulky(firstLine: string, size: number, filler: string): string {
	return `${firstLine}\n${filler.repeat(size)}`;
}

interface Fixture {
	session: ReturnType<typeof SessionManager.inMemory>;
	usage: { tokens: number; contextWindow: number; percent: number };
	ids: { biggestRead: string; bash: string };
}

function fixture(): Fixture {
	const session = SessionManager.inMemory();
	session.appendMessage(system("You are pi."));
	session.appendMessage(user("read the big file"));
	session.appendMessage(assistant("Reading now."));
	const biggestRead = session.appendMessage(toolResult("read", bulky("const foo = 1;", 4000, "A")));
	session.appendMessage(toolResult("read", bulky("const bar = 2;", 2000, "B")));
	const bash = session.appendMessage(toolResult("bash", bulky("$ make build", 1500, "C")));
	session.appendMessage(toolResult("read", bulky("const baz = 3;", 100, "D")));
	session.appendMessage(toolResult("task", bulky("subagent done", 3000, "E")));
	// Reported usage is an exact 2x of the char-based estimate, making calibration assertions exact.
	const est = session
		.buildSessionProjection()
		.messages.reduce((sum, m) => sum + estimateMessageTokens(m as Message), 0);
	return { session, usage: { tokens: 2 * est, contextWindow: 8000, percent: 66 }, ids: { biggestRead, bash } };
}

describe("context-map buildMap", () => {
	it("categorizes by role and tool name, promoting dominant unknown tools", () => {
		const f = fixture();
		const map = buildMap(f.session, f.usage, new Set());
		const keys = map.categories.map((c) => c.key);
		expect(keys).toContain("read");
		expect(keys).toContain("bash");
		expect(keys).toContain("user");
		expect(keys).toContain("assistant");
		// The only unknown tool dominates, so it is promoted and "other" disappears.
		expect(keys).toContain("tool:task");
		expect(keys).not.toContain("other");
		// Categories are sorted largest-first.
		expect(keys[0]).toBe("read");
		const read = map.categories.find((c) => c.key === "read");
		expect(read?.entries).toHaveLength(3);
		expect(read?.entries[0]?.label).toBe("const foo = 1;");
	});

	it("calibrates estimates against provider-reported usage", () => {
		const f = fixture();
		const map = buildMap(f.session, f.usage, new Set());
		expect(map.scale).toBeCloseTo(2, 5);
		expect(map.total).toBeCloseTo(f.usage.tokens, 0);
		expect(map.contextWindow).toBe(8000);
	});

	it("subtracts pending trims from the post-trim projection", () => {
		const f = fixture();
		const plain = buildMap(f.session, f.usage, new Set());
		const read = plain.categories.find((c) => c.key === "read");
		const trimmed = read?.entries[0];
		expect(trimmed?.id).toBe(f.ids.biggestRead);

		const after = buildMap(f.session, f.usage, new Set([f.ids.biggestRead]));
		expect(plain.total - after.total).toBeCloseTo(trimmed?.tokens ?? -1, 5);
		// The trimmed entry stays listed so the trim can still be undone.
		const readAfter = after.categories.find((c) => c.key === "read");
		expect(readAfter?.entries.some((e) => e.id === f.ids.biggestRead)).toBe(true);
	});

	it("honors earlier null context edits", () => {
		const f = fixture();
		f.session.appendContextEdit(f.ids.bash, null);
		const map = buildMap(f.session, f.usage, new Set());
		expect(map.categories.some((c) => c.key === "bash")).toBe(false);
	});
});

describe("context-map extension wiring", () => {
	const theme = { fg: (_token: string, text: string) => text } as unknown as Theme;

	function setup() {
		const handlers = new Map<string, (event: unknown, ctx: ExtensionContext) => unknown>();
		const commands = new Map<string, { handler: (args: string, ctx: ExtensionCommandContext) => Promise<void> }>();
		const pi = {
			on: (event: string, handler: (event: unknown, ctx: ExtensionContext) => unknown) => {
				handlers.set(event, handler);
				return () => handlers.delete(event);
			},
			registerCommand: (
				name: string,
				def: { handler: (args: string, ctx: ExtensionCommandContext) => Promise<void> },
			) => {
				commands.set(name, def);
			},
		} as unknown as ExtensionAPI;
		contextMapExtension(pi);
		return { handlers, commands };
	}

	function overlayCtx(f: Fixture, component: { box: ContextMapComponent | undefined }) {
		return {
			mode: "tui",
			sessionManager: f.session,
			getContextUsage: () => f.usage,
			ui: {
				notify: () => {},
				// Invoke the factory exactly as pi does, with close as the 4th argument;
				// a regression once shipped a no-op done that made Escape silently leak
				// the overlay open forever.
				custom: async (
					factory: (tui: unknown, thm: Theme, keybindings: unknown, done: () => void) => ContextMapComponent,
				) => {
					let closed = false;
					component.box = factory(undefined, theme, undefined, () => {
						closed = true;
					});
					expect(closed).toBe(false);
				},
			},
		} as unknown as ExtensionCommandContext;
	}

	it("registers the command, events, and stays idle without queued trims", async () => {
		const { handlers, commands } = setup();
		expect([...commands.keys()]).toEqual(["contextmap"]);
		expect(handlers.has("turn_end")).toBe(true);
		expect(handlers.has("session_start")).toBe(true);
		expect(handlers.has("session_shutdown")).toBe(true);
		const f = fixture();
		const ctx = { sessionManager: f.session } as unknown as ExtensionContext;
		const idle = (await handlers.get("turn_end")?.({ type: "turn_end" }, ctx)) as { entries?: unknown[] } | undefined;
		expect(idle).toBeUndefined();
	});

	it("drives a trim end to end: overlay toggle, turn_end draft, session projection", async () => {
		const { handlers, commands } = setup();
		const f = fixture();
		const component: { box: ContextMapComponent | undefined } = { box: undefined };
		const ctx = overlayCtx(f, component);

		await commands.get("contextmap")?.handler("", ctx);
		expect(component.box).toBeDefined();
		const comp = component.box as ContextMapComponent;

		// Expand the largest category (read), move to its largest entry, trim it.
		comp.handleInput("\r");
		comp.handleInput("\x1b[B");
		comp.handleInput("t");
		const overlay = comp.render(80).join("\n");
		expect(overlay).toContain("⌫1 pending");
		expect(overlay).toContain("const foo = 1;");

		const result = (await handlers.get("turn_end")?.({ type: "turn_end" }, ctx)) as {
			entries: { type: string; targetId: string; replacement: null }[];
		};
		expect(result?.entries).toEqual([{ type: "context_edit", targetId: f.ids.biggestRead, replacement: null }]);

		// The draft is exactly what SessionManager.appendContextEdit accepts; committing it
		// removes the tool result from the model projection while the transcript keeps it.
		for (const draft of result.entries) f.session.appendContextEdit(draft.targetId, draft.replacement);
		const projected = f.session.buildSessionProjection().messages;
		expect(
			projected.some(
				(m) => m.role === "toolResult" && m.content.some((b) => b.type === "text" && b.text.includes("const foo")),
			),
		).toBe(false);
		expect(f.session.getBranch().filter((e) => e.type === "message")).toHaveLength(8);
	});
});

describe("context-map component", () => {
	const theme = { fg: (_token: string, text: string) => text } as unknown as Theme;

	function component(f: Fixture, pending: Set<string>): ContextMapComponent {
		return new ContextMapComponent(
			theme,
			() => buildMap(f.session, f.usage, pending),
			pending,
			() => {},
		);
	}

	it("renders the map with legend and hints", () => {
		const f = fixture();
		const lines = component(f, new Set()).render(80);
		const text = lines.join("\n");
		expect(text).toContain("Context Map");
		expect(text).toContain("read");
		expect(text).toContain("task");
		expect(text).toContain("Esc close");
	});

	it("keeps the cursor on the same row across the trim refresh", () => {
		const f = fixture();
		const pending = new Set<string>();
		const comp = component(f, pending);

		comp.handleInput("\r"); // expand the first category (read, largest)
		comp.handleInput("\x1b[B"); // move to its largest entry
		const before = comp.render(80).find((line) => line.includes("▶"));
		expect(before).toContain("const foo = 1;");

		comp.handleInput("t"); // trim it
		expect(pending.size).toBe(1);
		const selected = comp.render(80).find((line) => line.includes("▶"));
		expect(selected).toContain("const foo = 1;");
		expect(comp.render(80).join("\n")).toContain("⌫1 pending");

		comp.handleInput("t"); // undo
		expect(pending.size).toBe(0);
		expect(comp.render(80).join("\n")).not.toContain("⌫1 pending");
	});

	it("closes on escape", () => {
		const f = fixture();
		let closed = false;
		const comp = new ContextMapComponent(
			theme,
			() => buildMap(f.session, f.usage, new Set()),
			new Set(),
			() => {
				closed = true;
			},
		);
		comp.handleInput("\x1b");
		expect(closed).toBe(true);
	});

	it("renders an empty session without throwing", () => {
		const session = SessionManager.inMemory();
		const comp = new ContextMapComponent(
			theme,
			() => buildMap(session, undefined, new Set()),
			new Set(),
			() => {},
		);
		expect(comp.render(80).join("\n")).toContain("Context Map");
	});
});
