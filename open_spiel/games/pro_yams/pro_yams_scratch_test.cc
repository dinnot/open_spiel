#include "open_spiel/games/pro_yams/pro_yams.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace pro_yams {
namespace {

void RetroactiveScratchingTest() {
  std::shared_ptr<const Game> game = LoadGame("pro_yams");
  std::unique_ptr<State> state = game->NewInitialState();
  ProYamsState* p_state = static_cast<ProYamsState*>(state.get());

  // 1. Initial Chance Node (Coefficients)
  state->ApplyAction(state->ChanceOutcomes()[0].first);

  // 2. Setup: Player 0 fills LS (Large Sum) in Free Column with a valid score.
  // P0 Turn 1.
  // Roll 6,6,6,6,6 (Index 7775).
  state->ApplyAction(7775);

  // Score LS in Free Column (Col 1).
  int action_ls = kActionScoreBase + (kColFree * kNumRows) + kRowLS;
  state->ApplyAction(action_ls);

  // 3. Skip P1 turn to get back to P0.
  // P1 Roll
  state->ApplyAction(0); // 1,1,1,1,1
  // P1 Score something
  int action_p1 = kActionScoreBase + (kColFree * kNumRows) + kRow1;
  state->ApplyAction(action_p1);

  // 4. P0 Turn 2.
  // Goal: Scratch SS (Small Sum).
  // We need a roll that is NOT a Yams (to avoid Sec bonus) and sum < 20.
  // Action 1 corresponds to 2,1,1,1,1 (Sum 6).
  state->ApplyAction(1);

  int action_ss = kActionScoreBase + (kColFree * kNumRows) + kRowSS; // 13 + 6 = 19.

  // Apply Scratch SS
  state->ApplyAction(action_ss);

  // 5. Verify Retroactive Scratching
  std::vector<float> obs_tensor = state->ObservationTensor(0);
  // P0, Col 1, Row 7 (LS) -> Index 20.

  float ls_val = obs_tensor[20];
  std::cout << "LS Value (normalized): " << ls_val << std::endl;

  if (std::abs(ls_val - 0.0f) < 1e-6) {
      std::cout << "TEST PASSED: LS was retroactively scratched." << std::endl;
  } else {
      std::cout << "TEST FAILED: LS was NOT scratched. Value: " << ls_val * 100 << std::endl;
      exit(1);
  }
}

}  // namespace
}  // namespace pro_yams
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::pro_yams::ForceProYamsLink();
  open_spiel::pro_yams::RetroactiveScratchingTest();
}
