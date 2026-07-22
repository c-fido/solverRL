#include <gtest/gtest.h>

#include "solverrl/vocabulary.hpp"

using solverrl::ArgSort;
using solverrl::Vocabulary;

TEST(KeyDoorVocabulary, ContainsPaperPerceptionAndDirToPredicates) {
  const auto vocab = Vocabulary::KeyDoor();
  EXPECT_EQ(vocab.name, "keydoor");

  ASSERT_NE(vocab.find("on_key"), nullptr);
  ASSERT_NE(vocab.find("door_open"), nullptr);
  ASSERT_NE(vocab.find("same_room_goal"), nullptr);
  ASSERT_NE(vocab.find("adj_door"), nullptr);
  ASSERT_NE(vocab.find("carrying"), nullptr);
  ASSERT_NE(vocab.find("dir_to"), nullptr);

  EXPECT_EQ(vocab.find("on_key")->arity(), 1u);
  EXPECT_EQ(vocab.find("dir_to")->arity(), 3u);
}

TEST(KeyDoorVocabulary, DirToBindsSharedDirectionVariable) {
  const auto vocab = Vocabulary::KeyDoor();
  const auto* dir_to = vocab.find("dir_to");
  ASSERT_NE(dir_to, nullptr);
  EXPECT_TRUE(dir_to->binds_direction_var);
  ASSERT_EQ(dir_to->arg_sorts.size(), 3u);
  EXPECT_EQ(dir_to->arg_sorts[0], ArgSort::State);
  EXPECT_EQ(dir_to->arg_sorts[1], ArgSort::Object);
  EXPECT_EQ(dir_to->arg_sorts[2], ArgSort::Direction);
}

TEST(KeyDoorVocabulary, ObjectAndDirectionDomains) {
  const auto vocab = Vocabulary::KeyDoor();
  EXPECT_EQ(vocab.objects, (std::vector<std::string>{"key", "door", "goal"}));
  EXPECT_EQ(vocab.directions,
            (std::vector<std::string>{"up", "down", "left", "right"}));
}

TEST(KeyDoorVocabulary, UnaryPerceptionPredicatesAllowNegation) {
  const auto vocab = Vocabulary::KeyDoor();
  for (const char* name :
       {"on_key", "door_open", "same_room_goal", "adj_door", "carrying"}) {
    const auto* p = vocab.find(name);
    ASSERT_NE(p, nullptr) << name;
    EXPECT_TRUE(p->allows_negation) << name;
    EXPECT_FALSE(p->binds_direction_var) << name;
    ASSERT_EQ(p->arg_sorts.size(), 1u);
    EXPECT_EQ(p->arg_sorts[0], ArgSort::State);
  }
}
