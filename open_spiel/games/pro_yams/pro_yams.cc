#include "open_spiel/games/pro_yams/pro_yams.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <iomanip>
#include <sstream>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace pro_yams {

// LINKER FIX: Dummy function definition
void ForceProYamsLink() {}

namespace {
const std::vector<int> kStandardCoeffs = {8, 10, 12, 14, 16, 18};

bool CheckBit(int mask, int bit) { return (mask & (1 << bit)) != 0; }

std::vector<int> DecodePermutation(int index, std::vector<int> items) {
  std::vector<int> result;
  std::vector<int> available = items;
  int n = items.size();
  int factorial = 1;
  for (int i = 2; i < n; ++i) factorial *= i; 

  for (int i = n - 1; i >= 0; --i) {
    int current_factorial = (i == 0) ? 1 : factorial;
    if (i > 0) factorial /= i;
    
    int selected_idx = index / current_factorial;
    index %= current_factorial;
    
    result.push_back(available[selected_idx]);
    available.erase(available.begin() + selected_idx);
  }
  return result;
}

const GameType kGameType{
    /*short_name=*/"pro_yams",
    /*long_name=*/"Pro Yams",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{},
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/false
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ProYamsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

ProYamsGame::ProYamsGame(const GameParameters& params)
    : Game(kGameType, params) {}

ProYamsState::ProYamsState(std::shared_ptr<const Game> game)
    : State(game), dice_(kNumDice, 0) {
  for (int p = 0; p < kNumPlayers; ++p) {
    for (int c = 0; c < kNumColumns; ++c) {
      for (int r = 0; r < kNumRows; ++r) {
        board_[p][c][r] = -1;
      }
    }
  }
}

Player ProYamsState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  return is_chance_node_ ? kChancePlayerId : current_player_;
}

std::vector<Action> ProYamsState::LegalActions() const {
  if (IsTerminal()) return {};
  if (is_chance_node_) return LegalChanceOutcomes();

  std::vector<Action> moves;

  // 1. Scoring Actions
  for (int c = 0; c < kNumColumns; ++c) {
    if (c == kColTurbo && rolls_in_turn_ > 2) continue;

    for (int r = 0; r < kNumRows; ++r) {
      if (CanPut(c, r, current_player_)) {
        moves.push_back(kActionScoreBase + (c * kNumRows) + r);
      }
    }
  }

  // 2. Reroll Actions
  if (rolls_in_turn_ < 3) {
    bool can_reroll = true;
    
    if (rolls_in_turn_ == 2) {
      bool has_non_turbo_moves = false;
      for (int c = 0; c < kNumColumns; ++c) {
        if (c == kColTurbo) continue;
        for (int r = 0; r < kNumRows; ++r) {
           if (CanPut(c, r, current_player_)) {
             has_non_turbo_moves = true;
             break;
           }
        }
        if (has_non_turbo_moves) break;
      }
      if (!has_non_turbo_moves) can_reroll = false;
    }

    if (can_reroll) {
      for (int i = 1; i < 32; ++i) {
        moves.push_back(kActionRerollBase + i);
      }
    }
  }

  return moves;
}

void ProYamsState::DoApplyAction(Action action_id) {
  if (is_chance_node_) {
    if (phase_ == 0) {
      ResolveChanceCoefficients(action_id);
    } else {
      ResolveRoll(action_id);
    }
    return;
  }

  if (action_id >= kActionRerollBase) {
    int mask = action_id - kActionRerollBase;
    for (int i = 0; i < kNumDice; ++i) {
      if (CheckBit(mask, i)) {
        dice_[i] = 0; 
      }
    }
    is_chance_node_ = true;
    return;
  } 
  
  int col = (action_id - kActionScoreBase) / kNumRows;
  int row = (action_id - kActionScoreBase) % kNumRows;

  int score = CalculateScore(col, row, dice_);
  board_[current_player_][col][row] = score;
  cells_filled_++;

  current_player_ = 1 - current_player_;
  rolls_in_turn_ = 0;
  std::fill(dice_.begin(), dice_.end(), 0); 
  is_chance_node_ = true; 
}

std::vector<std::pair<Action, double>> ProYamsState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;

  if (phase_ == 0) {
    double prob = 1.0 / 720.0;
    for (int i = 0; i < 720; ++i) {
      outcomes.push_back({i, prob});
    }
    return outcomes;
  }

  int num_to_roll = 0;
  for (int d : dice_) if (d == 0) num_to_roll++;

  int num_outcomes = std::pow(6, num_to_roll);
  double prob = 1.0 / num_outcomes;

  for (int i = 0; i < num_outcomes; ++i) {
    outcomes.push_back({i, prob});
  }
  return outcomes;
}

void ProYamsState::ResolveChanceCoefficients(Action action_id) {
  column_coefficients_ = DecodePermutation(action_id, kStandardCoeffs);
  phase_ = 1;
  rolls_in_turn_ = 0;
  current_player_ = 0;
  std::fill(dice_.begin(), dice_.end(), 0);
}

void ProYamsState::ResolveRoll(Action action_id) {
  int temp_id = action_id;
  for (int i = 0; i < kNumDice; ++i) {
    if (dice_[i] == 0) {
      dice_[i] = (temp_id % 6) + 1;
      temp_id /= 6;
    }
  }
  rolls_in_turn_++;
  is_chance_node_ = false;
}

bool ProYamsState::CanPut(int col, int row, Player p) const {
  if (board_[p][col][row] != -1) return false; 

  if (col == kColDown) { 
    if (row > 0 && board_[p][col][row - 1] == -1) return false;
  } else if (col == kColUp) { 
    if (row < kNumRows - 1 && board_[p][col][row + 1] == -1) return false;
  } else if (col == kColMid) { 
    if (row != 5 && row != 6) {
        bool has_upper = (row > 0 && board_[p][col][row - 1] != -1);
        bool has_lower = (row < kNumRows - 1 && board_[p][col][row + 1] != -1);
        if (!has_upper && !has_lower) return false;
    }
    if (row < 5 && board_[p][col][row + 1] == -1) return false;
    if (row > 6 && board_[p][col][row - 1] == -1) return false;
  } else if (col == kColUpDown) { 
    int count = 0;
    for (int r = 0; r < kNumRows; ++r) if (board_[p][col][r] != -1) count++;
    if (count > 0) {
      bool has_upper = (row > 0 && board_[p][col][row - 1] != -1);
      bool has_lower = (row < kNumRows - 1 && board_[p][col][row + 1] != -1);
      if (!has_upper && !has_lower) return false;
    }
  }
  return true;
}

int ProYamsState::CalculateScore(int col, int row, const std::vector<int>& dice) const {
  int raw_score = 0;
  bool is_sec = IsSecBonus();

  if (is_sec) {
    if (row <= kRow6) raw_score = (row + 1) * 5;
    else if (row == kRowSS) {
        int my_ls = board_[CurrentPlayer()][col][kRowLS];
        if (my_ls != -1 && my_ls > 0) raw_score = my_ls - 1;
        else raw_score = 29;
    }
    else if (row == kRowLS) raw_score = 30; 
    else if (row == kRowFH) raw_score = 50;
    else if (row == kRowK) raw_score = 54;
    else if (row == kRowQ) raw_score = 50;
    else if (row == kRow8) raw_score = 75; 
    else if (row == kRowY) raw_score = 100;
  } else {
      auto counts = GetDiceCounts();
      int sum = GetDiceSum();

      if (row <= kRow6) { 
        int target = row + 1;
        raw_score = counts[target] * target;
      } else {
          switch (row) {
            case kRowSS: 
            case kRowLS: 
              raw_score = sum; 
              break;
            case kRowFH: {
              bool has3 = false, has2 = false;
              for (int c : counts) {
                if (c == 3) has3 = true;
                if (c == 2) has2 = true;
                if (c == 5) { has3 = true; has2 = true; } 
              }
              raw_score = (has3 && has2) ? (20 + sum) : 0;
              break;
            }
            case kRowK: {
              bool has4 = false;
              for (int c : counts) if (c >= 4) has4 = true;
              raw_score = has4 ? (30 + sum) : 0;
              break;
            }
            case kRowQ: { 
              bool small = (counts[1] && counts[2] && counts[3] && counts[4] && counts[5]);
              bool large = (counts[2] && counts[3] && counts[4] && counts[5] && counts[6]);
              if (small) raw_score = 45;
              else if (large) raw_score = 50;
              else raw_score = 0;
              break;
            }
            case kRow8:
              raw_score = (sum <= 8) ? (60 + 5 * (8 - sum)) : 0;
              break;
            case kRowY: {
              bool yams = false;
              int face = 0;
              for (int i=1; i<=6; ++i) if(counts[i] == 5) { yams = true; face=i; }
              raw_score = yams ? (75 + 5 * (face - 1)) : 0;
              break;
            }
          }
      }
  }

  // --- Strict Zeroing Logic ---
  if (row == kRowLS && board_[CurrentPlayer()][col][kRowSS] == 0) {
    return 0;
  }
  
  if (!is_sec) {
      if ((row == kRowSS || row == kRowLS) && raw_score < 20) return 0;

      if (row == kRowSS) {
          int my_ls = board_[CurrentPlayer()][col][kRowLS];
          if (my_ls != -1 && raw_score >= my_ls) return 0;
          if (raw_score > 29) return 0; 
      }
      if (row == kRowLS) {
          int highest_ss = -1;
          for (int opp = 0; opp < kNumPlayers; ++opp) {
              if (board_[opp][col][kRowSS] > highest_ss) highest_ss = board_[opp][col][kRowSS];
          }
          if (highest_ss != -1 && raw_score <= highest_ss) return 0;
      }

      int max_existing = -1;
      for (int other_p = 0; other_p < kNumPlayers; ++other_p) {
        if (board_[other_p][col][row] > max_existing) {
          max_existing = board_[other_p][col][row];
        }
      }
      if (max_existing != -1 && raw_score < max_existing) return 0;
  }

  return raw_score;
}

int ProYamsState::GetDiceSum() const {
  return std::accumulate(dice_.begin(), dice_.end(), 0);
}

std::vector<int> ProYamsState::GetDiceCounts() const {
  std::vector<int> counts(7, 0);
  for (int d : dice_) counts[d]++;
  return counts;
}

bool ProYamsState::IsSecBonus() const {
  if (rolls_in_turn_ != 1) return false;
  auto counts = GetDiceCounts();
  for (int c : counts) if (c == 5) return true;
  return false;
}

bool ProYamsState::IsTerminal() const {
  return cells_filled_ >= (kNumPlayers * kNumCells);
}

std::vector<double> ProYamsState::Returns() const {
  if (!IsTerminal()) return {0.0, 0.0};

  double p0_score = 0;

  for (int c = 0; c < kNumColumns; ++c) {
    bool p0_clean = false;
    bool p1_clean = false;
    int raw0 = CalculateColumnScore(c, 0, p0_clean);
    int raw1 = CalculateColumnScore(c, 1, p1_clean);

    int multiplier = 1;
    if (raw1 > 0 && raw0 >= 2 * raw1) multiplier = std::min(5, raw0 / raw1);
    else if (raw0 > 0 && raw1 >= 2 * raw0) multiplier = std::min(5, raw1 / raw0);
    
    if (raw1 == 0 && raw0 > 0) multiplier = 5;
    if (raw0 == 0 && raw1 > 0) multiplier = 5;

    int diff = raw0 - raw1;
    if (p0_clean) diff += 200;
    if (p1_clean) diff -= 200;

    int col_points = diff * column_coefficients_[c] * multiplier;
    p0_score += col_points;
  }
  
  return {p0_score, -p0_score};
}

int ProYamsState::CalculateColumnScore(int col, Player p, bool& out_clean_bonus) const {
  int sum = 0;
  int upper_sum = 0;
  out_clean_bonus = true;

  for (int r = 0; r < kNumRows; ++r) {
    int val = board_[p][col][r];
    if (val == 0) out_clean_bonus = false;
    if (val > 0) sum += val;
    if (r <= kRow6 && val > 0) upper_sum += val;
  }

  if (upper_sum >= 100) sum += 500;
  else if (upper_sum >= 90) sum += 200;
  else if (upper_sum >= 80) sum += 100;
  else if (upper_sum >= 70) sum += 50;
  else if (upper_sum >= 60) sum += 30;

  return sum;
}

std::string ProYamsState::ToString() const {
  std::stringstream ss;
  ss << "Phase: " << phase_ << "\n";
  ss << "Coeffs: ";
  for (int c : column_coefficients_) ss << c << " ";
  ss << "\nDice: ";
  for (int d : dice_) ss << d << " ";
  ss << "\nRolls: " << rolls_in_turn_ << "\n";
  return ss.str();
}

std::string ProYamsState::ActionToString(Player player, Action action_id) const {
  if (player == kChancePlayerId) {
    std::stringstream ss;
    if (phase_ == 0) {
      ss << "Chance (Coeffs): " << action_id;
    } else {
      ss << "Chance (Roll): " << action_id;
    }
    return ss.str();
  }

  if (action_id >= kActionRerollBase) {
    int mask = action_id - kActionRerollBase;
    std::stringstream ss;
    ss << "Reroll dice indices: [";
    for (int i = 0; i < kNumDice; ++i) {
      if (CheckBit(mask, i)) {
        ss << i << " ";
      }
    }
    ss << "]";
    return ss.str();
  }

  int col = action_id / kNumRows;
  int row = action_id % kNumRows;

  const std::vector<std::string> col_names = {"Down", "Free", "Up", "Mid", "Turbo", "UpDown"};
  const std::vector<std::string> row_names = {"1", "2", "3", "4", "5", "6", "SS", "LS", "FH", "K", "Q", "8", "Y"};

  std::stringstream ss;
  ss << "Score " << col_names[col] << " -> " << row_names[row];
  return ss.str();
}

std::string ProYamsState::InformationStateString(Player player) const {
  return ToString();
}

std::string ProYamsState::ObservationString(Player player) const {
  return ToString();
}

void ProYamsState::ObservationTensor(Player player, absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0.0);
  int offset = 0;

  for (int p_offset = 0; p_offset < kNumPlayers; ++p_offset) {
    Player p = (player + p_offset) % kNumPlayers;
    for (int c = 0; c < kNumColumns; ++c) {
      for (int r = 0; r < kNumRows; ++r) {
        if (board_[p][c][r] != -1) {
            values[offset] = static_cast<float>(board_[p][c][r]) / 100.0f;
        } else {
            values[offset] = -1.0f; 
        }
        offset++;
      }
    }
  }

  for (int i = 0; i < kNumDice; ++i) {
    values[offset++] = static_cast<float>(dice_[i]) / 6.0f;
  }

  if (!column_coefficients_.empty()) {
      for (int i = 0; i < kNumColumns; ++i) {
        values[offset++] = static_cast<float>(column_coefficients_[i]) / 20.0f; 
      }
  } else {
      offset += kNumColumns;
  }

  values[offset++] = static_cast<float>(rolls_in_turn_) / 3.0f;
  values[offset++] = static_cast<float>(phase_);
  values[offset++] = (current_player_ == player) ? 1.0f : 0.0f;
}

std::unique_ptr<State> ProYamsState::Clone() const {
  return std::unique_ptr<State>(new ProYamsState(*this));
}

}  // namespace pro_yams
}  // namespace open_spiel