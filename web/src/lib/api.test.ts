import { describe, expect, it } from "vitest";
import { matchesName } from "./api";

describe("matchesName", () => {
  const watkins = { web_name: "Watkins", canonical_name: "Ollie Watkins" };

  it("matches the full name people actually type, not just the shirt name", () => {
    expect(matchesName(watkins, "ollie")).toBe(true);   // web_name alone would miss this
    expect(matchesName(watkins, "watkins")).toBe(true);
    expect(matchesName(watkins, "haaland")).toBe(false);
  });

  it("shows everyone when the box is empty or whitespace", () => {
    expect(matchesName(watkins, "")).toBe(true);
    expect(matchesName(watkins, "   ")).toBe(true);
  });

  it("survives a player with no canonical name on file", () => {
    expect(matchesName({ web_name: "Raya" }, "raya")).toBe(true);
    expect(matchesName({ web_name: "Raya" }, "ollie")).toBe(false);
  });
});
