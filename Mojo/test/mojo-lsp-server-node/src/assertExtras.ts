import * as assert from "assert";

export const contains = (actual: string, substr: string) => {
  if (actual.indexOf(substr) === -1) {
    assert.fail(`expected '${actual}' to contain '${substr}'`);
  }
};

export const doesNotContain = (actual: string, substr: string) => {
  if (actual.indexOf(substr) !== -1) {
    assert.fail(`expected '${actual}' to not contain '${substr}'`)
  }
}
