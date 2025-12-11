// open_spiel/games/pro_yams/pro_yams.h
#ifndef OPEN_SPIEL_GAMES_PRO_YAMS_H_
#define OPEN_SPIEL_GAMES_PRO_YAMS_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace pro_yams {

constexpr int kNumPlayers = 2;
constexpr int kNumDice = 5;
constexpr int kNumDieSides = 6;
constexpr int kNumRows = 13;
constexpr int kNumColumns = 6;
constexpr int kNumCells = kNumRows * kNumColumns;

// Actions: 0-77 (Score), 78-109 (Reroll bitmasks)
constexpr int kActionScoreBase = 0; 
constexpr int kActionRerollBase = 78; 
constexpr int kNumDistinctActions = kActionRerollBase + 32; 

enum ColumnType {
  kColDown = 0,
  kColFree = 1,
  kColUp = 2,
  kColMid = 3,
  kColTurbo = 4,   
  kColUpDown = 5   
};

enum RowType {
  kRow1 = 0, kRow2, kRow3, kRow4, kRow5, kRow6,
  kRowSS, kRowLS, kRowFH, kRowK, kRowQ, kRow8, kRowY
};

class ProYamsGame;

class ProYamsState : public State {
 public:
  ProYamsState(std::shared_ptr<const Game> game);
  ProYamsState(const ProYamsState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  bool CanPut(int col, int row, Player p) const;
  int CalculateScore(int col, int row, const std::vector<int>& dice) const;
  int CalculateColumnScore(int col, Player p, bool& out_clean_bonus) const;
  
  void ResolveChanceCoefficients(Action action_id);
  void ResolveRoll(Action action_id);
  
  int GetDiceSum() const;
  std::vector<int> GetDiceCounts() const;
  bool IsSecBonus() const;

  Player current_player_ = kChancePlayer;
  int turn_ = 0;
  int rolls_in_turn_ = 0; 
  bool is_chance_node_ = true;
  int phase_ = 0; 

  std::vector<int> column_coefficients_; 
  std::vector<int> dice_; 
  
  // Board: [Player][Column][Row]. -1 indicates empty.
  std::array<std::array<std::array<int, kNumRows>, kNumColumns>, kNumPlayers> board_;
  int cells_filled_ = 0;
};

class ProYamsGame : public Game {
 public:
  explicit ProYamsGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ProYamsState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override { return 7776; } // 6^5
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -5000; } 
  double MaxUtility() const override { return 5000; }
  int MaxGameLength() const override { return kNumCells * kNumPlayers * 5; } 
  std::vector<int> ObservationTensorShape() const override {
    // Board (2*6*13) + Dice (5) + Coeffs (6) + Context (3)
    return {kNumPlayers * kNumColumns * kNumRows + kNumDice + kNumColumns + 3};
  }
};

}  // namespace pro_yams
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PRO_YAMS_H_
