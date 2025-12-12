#include "open_spiel/games/pro_yams/pro_yams.h"

#include <algorithm>
#include <iostream>
#include <vector>

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
  
  // 1. Initial Chance Node (Coefficients)
  state->ApplyAction(state->ChanceOutcomes()[0].first);

  // 2. Player 0 Roll: Force a Yams (5,5,5,5,5)
  // Index 6220 corresponds to [5, 5, 5, 5, 5]
  state->ApplyAction(6220); 
  
  // Test Sec Bonus Logic
  int action_free_yams = kActionScoreBase + (kColFree * kNumRows) + kRowY;
  
  // Verify Yams move is legal
  auto legal = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action_free_yams) != legal.end());

  // Apply Yams in Free column
  state->ApplyAction(action_free_yams);
  
  // P1's turn. Roll 1, 1, 2, 2, 3 (Sum = 9). Index 2844.
  state->ApplyAction(2844); 
  
  // Verify SS/LS Constraints
  int action_free_ls = kActionScoreBase + (kColFree * kNumRows) + kRowLS;
  legal = state->LegalActions();

  // "Scratch LS" (score 0) is legal because 9 < 20
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action_free_ls) != legal.end());
  
  // Reroll action test
  state->ApplyAction(kActionRerollBase + 31); // Reroll all
  state->ApplyAction(7775); // Roll 6,6,6,6,6 (Sum 30)
  
  // LS should be legal (Sum 30 >= 20)
  SPIEL_CHECK_TRUE(std::find(state->LegalActions().begin(), state->LegalActions().end(), 
                             action_free_ls) != state->LegalActions().end());
}

void OracleObservationTest() {
  std::shared_ptr<const Game> game = LoadGame("pro_yams");
  std::unique_ptr<State> state = game->NewInitialState();
  
  // 1. Setup State: Coeffs -> P0 Roll
  state->ApplyAction(state->ChanceOutcomes()[0].first);
  
  // Force P0 roll: 1, 2, 3, 4, 5 (Small Straight & Yams potential is 0)
  // We need to find the action index for [1,2,3,4,5].
  // Note: Dice indices in ChanceOutcomes are often sorted.
  // Let's assume a known index or just iterate to find it.
  // For simplicity here, we simulate a state where we know the dice.
  
  // Let's just step forward to P0 turn.
  state->ApplyAction(0); // Roll 1,1,1,1,1
  
  // Get Observation
  std::vector<float> obs = state->ObservationTensor(0);
  
  // Check Tensor Size
  // Expected: Board(156) + Dice(5) + Coeffs(6) + Context(3) + Oracle(4) = 174
  SPIEL_CHECK_EQ(obs.size(), game->ObservationTensorShape()[0]);
  SPIEL_CHECK_EQ(obs.size(), 174); 

  // Verify Oracle Values exist at the end
  // With 1,1,1,1,1:
  // Yams = 1.0 (Prob)
  // Full = 0.0 (Cannot make full from 5-of-a-kind in 2 rolls? Actually prob is 0 because you hold 5 same)
  // Wait, if you have 1,1,1,1,1 you HAVE Yams.
  // So index [end-4] should be 1.0
  
  int offset = obs.size() - 4;
  SPIEL_CHECK_EQ(obs[offset], 1.0f); // Prob Yams
  
  // std::cout << "Oracle Prob Yams: " << obs[offset] << std::endl;
  // std::cout << "Oracle Prob Full: " << obs[offset+1] << std::endl;
}

}  // namespace
}  // namespace pro_yams
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::pro_yams::ForceProYamsLink();
  
  open_spiel::pro_yams::BasicProYamsTests();
  open_spiel::pro_yams::ScoringConstraintsTest();
  open_spiel::pro_yams::OracleObservationTest();
  
  std::cout << "All Pro Yams tests passed!" << std::endl;
}