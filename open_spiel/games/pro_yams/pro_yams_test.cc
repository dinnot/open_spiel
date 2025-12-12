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
  // Just pick the first outcome 
  state->ApplyAction(state->ChanceOutcomes()[0].first);

  // 2. Player 0 Roll
  // Force a Yams (5,5,5,5,5) to test Sec Bonus
  // Index 6220 corresponds to [5, 5, 5, 5, 5]
  state->ApplyAction(6220); 
  
  // Test Sec Bonus Logic
  // With Sec (Yams on first roll), we can fill ANY available cell.
  // Let's check a Free column (col 1).
  int action_free_yams = kActionScoreBase + (kColFree * kNumRows) + kRowY;
  SPIEL_CHECK_TRUE(std::find(state->LegalActions().begin(), state->LegalActions().end(), 
                             action_free_yams) != state->LegalActions().end());

  // Apply Yams in Free column
  state->ApplyAction(action_free_yams);
  
  // Now it's P1's turn. 
  // P1 Rolls. Let's give a non-Sec roll: 1, 1, 2, 2, 3 (Sum = 9)
  // Index 2844 corresponds to [0, 0, 1, 1, 2] (dice 1,1,2,2,3)
  state->ApplyAction(2844); 
  
  // Verify SS/LS Constraints logic
  // Try to score LS (Large Sum) in Free Column for P1.
  int action_free_ls = kActionScoreBase + (kColFree * kNumRows) + kRowLS;
  auto legal = state->LegalActions();

  // Since we relaxed CanPut to allow scratching (scoring 0), this move IS legal now.
  // It effectively means "Scratch LS" because 9 < 20.
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action_free_ls) != legal.end());
  
  // However, players usually want to reroll to get a better score.
  // Reroll the 3 (die index 4). Mask = 1<<4 = 16.
  // Action = kActionRerollBase + 16 = 78 + 16 = 94.
  state->ApplyAction(94);
  
  // Chance node for 1 die. Roll a 6. (Index 5).
  state->ApplyAction(5);
  // Dice: 1, 1, 2, 2, 6. Sum = 12. Still < 20. 
  
  // Reroll everything (Mask 31).
  state->ApplyAction(kActionRerollBase + 31);
  
  // Chance node 5 dice. Roll 6,6,6,6,6.
  // Index 7775 (5,5,5,5,5).
  state->ApplyAction(7775);
  // Dice: 6,6,6,6,6. Sum 30.
  // LS should definitely be legal (Sum 30 >= 20).
  SPIEL_CHECK_TRUE(std::find(state->LegalActions().begin(), state->LegalActions().end(), 
                             action_free_ls) != state->LegalActions().end());
}

}  // namespace
}  // namespace pro_yams
}  // namespace open_spiel

int main(int argc, char** argv) {
  // Force link the library
  open_spiel::pro_yams::ForceProYamsLink();

  open_spiel::pro_yams::BasicProYamsTests();
  open_spiel::pro_yams::ScoringConstraintsTest();
}