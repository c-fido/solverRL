#include <gtest/gtest.h>

#include <vector>

#include "solverrl/keydoor_ground.hpp"
#include "solverrl/prolog_emit.hpp"
#include "solverrl/rule_learner.hpp"

using solverrl::AtomSchema;
using solverrl::EmitConfig;
using solverrl::Example;
using solverrl::RuleLearner;

namespace {

EmitConfig synthetic_emit_config() {
  EmitConfig cfg;
  cfg.action_names = {"act0", "act1", "act2"};
  cfg.atoms = {
      AtomSchema{"atom0", {}, false, ""},
      AtomSchema{"atom1", {}, false, ""},
      AtomSchema{"atom2", {}, false, ""},
  };
  return cfg;
}

std::vector<Example> synthetic_planted() {
  std::vector<Example> data;
  for (int mask = 0; mask < 8; ++mask) {
    Example ex;
    ex.atoms = {(mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0};
    if (ex.atoms[0]) {
      ex.action = 0;
    } else if (ex.atoms[1]) {
      ex.action = 1;
    } else {
      ex.action = 2;
    }
    ex.weight = 1.0;
    data.push_back(ex);
  }
  return data;
}

}  // namespace

TEST(RuleLearner, KeyDoorFactoryStartsUnfitted) {
  const auto rl = RuleLearner::keydoor();
  EXPECT_FALSE(rl.is_fitted());
}

TEST(RuleLearner, FitPredictFidelityAndPrologOnSynthetic) {
  const auto data = synthetic_planted();
  RuleLearner learner(3, 3, synthetic_emit_config());
  learner.fit(data);
  EXPECT_TRUE(learner.is_fitted());
  EXPECT_GE(learner.n_clauses(), 1u);
  EXPECT_GE(learner.fidelity(data), 0.99);

  const std::string pl = learner.to_prolog();
  EXPECT_NE(pl.find("decision_list"), std::string::npos);
  EXPECT_NE(pl.rfind("act(_S,"), std::string::npos);

  for (const auto& ex : data) {
    EXPECT_EQ(learner.predict(ex), ex.action);
  }
}

TEST(RuleLearner, KeydoorFitOnSingleGroundedExample) {
  solverrl::KeyDoorState s;
  s.door_row = 0;
  s.agent_r = 0;
  s.agent_c = 0;
  s.key_r = 0;
  s.key_c = 0;
  s.goal_r = 0;
  s.goal_c = 4;
  s.door_open = false;
  s.carrying = false;

  Example ex;
  ex.atoms = solverrl::ground_atoms(s);
  ex.action = 4;  // pickup
  ex.weight = 1.0;

  auto rl = RuleLearner::keydoor();
  rl.fit({ex});
  EXPECT_EQ(rl.predict(ex), 4);
  EXPECT_DOUBLE_EQ(rl.fidelity({ex}), 1.0);
  EXPECT_FALSE(rl.to_prolog().empty());
}
