#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "solverrl/foil.hpp"
#include "solverrl/prolog_emit.hpp"
#include "solverrl/vocabulary.hpp"

using solverrl::AtomSchema;
using solverrl::Clause;
using solverrl::DecisionList;
using solverrl::EmitConfig;
using solverrl::Example;
using solverrl::Literal;
using solverrl::OrderedCovering;
using solverrl::Vocabulary;
using solverrl::emit_prolog;

namespace {

// Synthetic ground vocabulary (3 unary atoms) with a planted decision list:
//   atom0 → action 0
//   ¬atom0 ∧ atom1 → action 1
//   default → action 2
DecisionList planted_rules() {
  DecisionList list;
  Clause c0;
  c0.body = {Literal{0, false}};
  c0.head_action = 0;
  list.clauses.push_back(c0);

  Clause c1;
  c1.body = {Literal{0, true}, Literal{1, false}};
  c1.head_action = 1;
  list.clauses.push_back(c1);

  Clause def;
  def.is_default = true;
  def.head_action = 2;
  list.clauses.push_back(def);
  return list;
}

std::vector<Example> enumerate_from_planted(const DecisionList& planted) {
  std::vector<Example> data;
  // All 2^3 truth assignments; weight 1.
  for (int mask = 0; mask < 8; ++mask) {
    Example ex;
    ex.atoms = {(mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0};
    ex.action = planted.predict(ex);
    ex.weight = 1.0;
    data.push_back(ex);
  }
  return data;
}

EmitConfig synthetic_emit_config() {
  EmitConfig cfg;
  cfg.action_names = {"act0", "act1", "act2"};
  cfg.atoms = {
      AtomSchema{"atom0", {}, false},
      AtomSchema{"atom1", {}, false},
      AtomSchema{"atom2", {}, false},
  };
  return cfg;
}

}  // namespace

TEST(SyntheticVocabulary, KeyDoorVocabularyIsAvailableAsNamedSchema) {
  // Sanity: the real KeyDoor relational vocab is the production schema.
  const auto vocab = Vocabulary::KeyDoor();
  EXPECT_NE(vocab.find("dir_to"), nullptr);
  EXPECT_NE(vocab.find("on_key"), nullptr);
}

TEST(SyntheticVocabulary, RecoversPlantedDecisionListPredictions) {
  const DecisionList planted = planted_rules();
  const auto data = enumerate_from_planted(planted);

  OrderedCovering learner(/*n_atoms=*/3, /*n_actions=*/3, /*min_score=*/1e-12,
                          /*max_body_literals=*/3);
  const DecisionList learned = learner.fit(data);

  ASSERT_FALSE(learned.clauses.empty());
  EXPECT_TRUE(learned.clauses.back().is_default);

  for (const auto& ex : data) {
    EXPECT_EQ(learned.predict(ex), planted.predict(ex))
        << "atoms=[" << ex.atoms[0] << "," << ex.atoms[1] << "," << ex.atoms[2] << "]";
  }
}

TEST(SyntheticVocabulary, EmittedPrologMatchesKnownRuleShape) {
  const DecisionList planted = planted_rules();
  const std::string pl = emit_prolog(planted, synthetic_emit_config());

  EXPECT_NE(pl.find("perception_and_path_layer"), std::string::npos);
  EXPECT_NE(pl.find("decision_list"), std::string::npos);
  EXPECT_NE(pl.find("act(S, act0) :- atom0(S), !."), std::string::npos);
  EXPECT_NE(pl.find("act(S, act1) :- \\+ atom0(S), atom1(S), !."), std::string::npos);
  EXPECT_NE(pl.find("act(_S, act2)."), std::string::npos);

  // Default must be the last act/2 clause.
  const auto last_act = pl.rfind("act(");
  const auto def_pos = pl.rfind("act(_S, act2).");
  ASSERT_NE(last_act, std::string::npos);
  ASSERT_NE(def_pos, std::string::npos);
  EXPECT_EQ(def_pos, last_act);
}

TEST(SyntheticVocabulary, LearnedListEmitsValidCutProgram) {
  const DecisionList planted = planted_rules();
  const auto data = enumerate_from_planted(planted);
  OrderedCovering learner(3, 3);
  const DecisionList learned = learner.fit(data);
  const std::string pl = emit_prolog(learned, synthetic_emit_config());

  EXPECT_NE(pl.find("!."), std::string::npos);
  EXPECT_NE(pl.rfind("act(_S,"), std::string::npos);
  // Every training label still matches after learn → emit at the C++ level.
  for (const auto& ex : data) {
    EXPECT_EQ(learned.predict(ex), planted.predict(ex));
  }
}
