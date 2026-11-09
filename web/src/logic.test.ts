import { expect, test } from "bun:test";
import { takeFrames } from "./api";

test("takeFrames keeps a partial frame for the next read and reads event names", () => {
  const [frames, rest] = takeFrames('event: lair.skill\ndata: {"id":"3d"}\n\ndata: {"a":1}\n\ndata: [DO');
  expect(frames).toEqual([
    { event: "lair.skill", data: '{"id":"3d"}' },
    { event: "message", data: '{"a":1}' },
  ]);
  const [more] = takeFrames(rest + "NE]\n\n");
  expect(more).toEqual([{ event: "message", data: "[DONE]" }]);
});

import { mentionAt } from "./conversations/mention";

test("mentionAt finds the @query being typed before the caret", () => {
  expect(mentionAt("veja @pão", 9)).toEqual({ start: 5, query: "pão" });
  expect(mentionAt("@", 1)).toEqual({ start: 0, query: "" });
  expect(mentionAt("email a@b.com", 13)).toBeNull();
  expect(mentionAt("veja @pão e mais", 16)).toBeNull();
  expect(mentionAt("veja @pão e mais", 9)).toEqual({ start: 5, query: "pão" });
});
