#include <gtest/gtest.h>

#include <string>

#include "solverrl/prolog_emit.hpp"

using solverrl::AtomSchema;
using solverrl::Clause;
using solverrl::DecisionList;
using solverrl::EmitConfig;
using solverrl::Literal;
using solverrl::emit_prolog;

static EmitConfig tiny_config() {
  EmitConfig cfg;
  cfg.action_names = {"pickup", "toggle"};
  cfg.atoms = {
      AtomSchema{"on_key", {}, false},
      AtomSchema{"door_open", {}, false},
  };
  return cfg;
}

TEST(PrologEmit, NonDefaultClausesUseCutAndDefaultIsLast) {
  DecisionList list;
  Clause c0;
  c0.body = {Literal{0, false}, Literal{1, true}};
  c0.head_action = 0;
  list.clauses.push_back(c0);
  Clause def;
  def.is_default = true;
  def.head_action = 1;
  list.clauses.push_back(def);

  const std::string pl = emit_prolog(list, tiny_config());

  EXPECT_NE(pl.find(":-"), std::string::npos);
  EXPECT_NE(pl.find("!"), std::string::npos);
  EXPECT_NE(pl.find("on_key(S)"), std::string::npos);
  EXPECT_NE(pl.find("\\+ door_open(S)"), std::string::npos);
  EXPECT_NE(pl.find("act(S, pickup)"), std::string::npos);

  // Default last: unconditional act for toggle appears after the cut clause.
  const auto cut_pos = pl.rfind("!");
  const auto def_pos = pl.rfind("act(_S, toggle).");
  ASSERT_NE(cut_pos, std::string::npos);
  ASSERT_NE(def_pos, std::string::npos);
  EXPECT_GT(def_pos, cut_pos);
}

TEST(PrologEmit, IncludesPerceptionAndPathLayerWithDirTo) {
  DecisionList list;
  Clause def;
  def.is_default = true;
  def.head_action = 0;
  list.clauses.push_back(def);

  EmitConfig cfg = tiny_config();
  cfg.atoms.push_back(AtomSchema{"dir_to", {"goal"}, true});

  const std::string pl = emit_prolog(list, cfg);
  EXPECT_NE(pl.find("perception_and_path_layer"), std::string::npos);
  EXPECT_NE(pl.find("dir_to("), std::string::npos);
  EXPECT_NE(pl.find("decision_list"), std::string::npos);
}

TEST(PrologEmit, DirToLiteralSharesDirectionVariableInHead) {
  DecisionList list;
  Clause c;
  c.body = {Literal{0, false}};
  c.head_action = 0;  // will map to move(D) when action name uses D
  list.clauses.push_back(c);
  Clause def;
  def.is_default = true;
  def.head_action = 0;
  list.clauses.push_back(def);

  EmitConfig cfg;
  cfg.action_names = {"move(D)"};
  cfg.atoms = {AtomSchema{"dir_to", {"key"}, true}};

  const std::string pl = emit_prolog(list, cfg);
  EXPECT_NE(pl.find("act(S, move(D)) :- dir_to(S, key, D), !."), std::string::npos);
}
