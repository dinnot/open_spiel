// open_spiel/games/pro_yams/pro_yams_test.cc
#include "open_spiel/games/pro_yams/pro_yams.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace pro_yams {
namespace {

namespace testing = open_spiel::testing;

void BasicProYamsTests() {
  testing::LoadGameTest("pro_yams");
  testing::RandomSimTest(*LoadGame("pro_yams"), 10);
}

void ScoringConstraintsTest() {
  std::shared_ptr<const Game> game = LoadGame("pro_yams");
  std::unique_ptr<State> state = game->NewInitialState();
  ProYamsState* p_state = static_cast<ProYamsState*>(state.get());

  // 1. Initial Chance Node (Coefficients)
  // Just pick the first outcome (sorted coeffs usually, or specific perm)
  state->ApplyAction(state->ChanceOutcomes()[0].first);

  // 2. Player 0 Roll (Force a specific roll for testing)
  // Let's force a roll that is NOT a Yams (to avoid Sec bonus for now)
  // Roll: 1, 2, 3, 4, 5 (Small Straight)
  // Action encoding: 0 (1,1,1,1,1) ... need to calculate index for 1,2,3,4,5
  // Actually, random sim handles generic moves. Let's just ApplyAction based on logic.
  
  // We can hack the state for testing logic directly if we exposed helpers, 
  // but strictly via API we must navigate chance nodes.
  
  // Navigate to P0 turn
  // P0 rolls. Let's assume the chance outcome gives us dice=[5,5,5,5,5] (Yams)
  // This is action index corresponding to 5,5,5,5,5. 
  // 6^0*4 + 6^1*4 + 6^2*4 + 6^3*4 + 6^4*4 = 4*(1+6+36+216+1296) = 4*1555 = 6220
  // Note: internal dice are 1-based, action math is 0-based. 5->4.
  
  // Test Sec Bonus Logic
  // First roll of the turn (rolls_in_turn = 0 -> becomes 1)
  state->ApplyAction(6220); // Dice are now [5, 5, 5, 5, 5]
  
  // Check legal actions. With Sec (Yams on first roll), we should be able to score 
  // almost anywhere. 
  // Let's verify we can score Yams (row 12) in 'Down' column (col 0).
  // Action = kActionScoreBase + col*13 + row
  int action_down_yams = kActionScoreBase + (kColDown * kNumRows) + kRowY;
  
  // This should be legal because Sec allows filling any available cell 
  // (subject to column structure?). 
  // Actually "Down" must be filled top-to-bottom. 
  // Sec bonus: "acts as a Joker, allowing you to fill ANY available cell... with max score".
  // Does it override structure? "in the column you are playing". 
  // Usually "available" implies structure must be respected in Yams strategy games, 
  // but "Joker" suggests flexibility. 
  // Your provided text says: "fill ANY available cell". 
  // In `pro_yams.cc`, `CanPut` respects structure. 
  // If `Down` requires row 0 first, row 12 is NOT available yet.
  
  // Let's check a Free column (col 1).
  int action_free_yams = kActionScoreBase + (kColFree * kNumRows) + kRowY;
  SPIEL_CHECK_TRUE(std::find(state->LegalActions().begin(), state->LegalActions().end(), 
                             action_free_yams) != state->LegalActions().end());

  // Apply Yams in Free column
  state->ApplyAction(action_free_yams);
  
  // Now it's P1's turn. 
  // P1 Rolls. Let's give a non-Sec roll: 1, 1, 2, 2, 3
  // 0, 0, 1, 1, 2. Index = 0 + 0 + 36 + 216 + 2*1296 = 252 + 2592 = 2844
  state->ApplyAction(2844); 
  
  // Verify SS/LS Constraints logic via `CanPut` checks implicitly
  // We can't verify protected methods directly without friend class or exposure.
  // But we can verify LegalActions.
  
  // Try to score LS (Large Sum) in Free Column for P1.
  // Requirement: Must be > highest SS recorded (currently 0/none).
  // Requirement: Must be >= 20.
  // Our roll sum is 1+1+2+2+3 = 9. 
  // 9 < 20. So LS should NOT be legal.
  int action_free_ls = kActionScoreBase + (kColFree * kNumRows) + kRowLS;
  auto legal = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action_free_ls) == legal.end());
  
  // Reroll to get higher score.
  // Reroll the 3 (die index 4). Mask = 1<<4 = 16.
  // Action = kActionRerollBase + 16 = 78 + 16 = 94.
  state->ApplyAction(94);
  
  // Chance node for 1 die. 
  // Roll a 6. (Index 5).
  state->ApplyAction(5);
  // Dice: 1, 1, 2, 2, 6. Sum = 12. Still < 20. LS illegal.
  
  // Reroll everything (Mask 31).
  state->ApplyAction(kActionRerollBase + 31);
  
  // Chance node 5 dice. Roll 6,6,6,6,6.
  // Index 7775 (5,5,5,5,5).
  state->ApplyAction(7775);
  // Dice: 6,6,6,6,6. Sum 30.
  // LS should now be legal (Sum 30 >= 20).
  SPIEL_CHECK_TRUE(std::find(state->LegalActions().begin(), state->LegalActions().end(), 
                             action_free_ls) != state->LegalActions().end());
}

}  // namespace
}  // namespace pro_yams
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::pro_yams::BasicProYamsTests();
  open_spiel::pro_yams::ScoringConstraintsTest();
}
