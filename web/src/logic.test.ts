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
