const { formatLatency } = require('../frontend/src/utils/formatLatency.js');

const cases = [
  { input: 842.1, expected: "0.84 sec" },
  { input: 1840.2, expected: "1.84 sec" },
  { input: 59990, expected: "59.99 sec" },
  { input: 60000, expected: "1 min 00 sec" },
  { input: 75432, expected: "1 min 15 sec" },
  { input: 125000, expected: "2 min 05 sec" },
  { input: null, expected: "N/A" },
  { input: undefined, expected: "N/A" },
  { input: "invalid", expected: "N/A" }
];

let allPassed = true;
console.log("=== TESTING formatLatency() UTILITY ===");
cases.forEach(({ input, expected }) => {
  const result = formatLatency(input);
  const pass = result === expected;
  if (!pass) allPassed = false;
  console.log(`formatLatency(${input}) => "${result}" [Expected: "${expected}"] - ${pass ? "PASS" : "FAIL"}`);
});

if (!allPassed) {
  process.exit(1);
} else {
  console.log("ALL formatLatency() TESTS PASSED!");
}
