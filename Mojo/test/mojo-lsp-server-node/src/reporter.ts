import * as Mocha from 'mocha';

class ModularReporter {
  constructor(runner: Mocha.Runner) {
    const _specReporter = new Mocha.reporters.Spec(runner);
    const _xunitReporter = new Mocha.reporters.XUnit(runner, {
      reporterOptions: {
        output: process.env["MOCHA_FILE"],
      },
    });
  }
}

module.exports = ModularReporter;
