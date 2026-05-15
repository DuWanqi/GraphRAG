from src.evaluation.factscore_adapter import (
    filter_historical_background_facts,
    is_historical_background_fact,
)
from src.evaluation.safe_checker import SAFEFactChecker


def test_historical_background_filter_keeps_public_context_only():
    facts = [
        "2009年全球金融危机影响中国外贸出口",
        "广州于2010年举办亚运会",
        "我每天早上挤地铁去白云区上班",
        "出租屋楼道狭窄，风扇吱呀转",
        "我心里始终留着一点不服输的劲",
    ]

    assert filter_historical_background_facts(facts) == [
        "2009年全球金融危机影响中国外贸出口",
        "广州于2010年举办亚运会",
    ]


def test_historical_background_filter_rejects_personal_narrative_with_dates():
    assert not is_historical_background_fact("2009年春天，我背着旧行李箱来到广州")
    assert not is_historical_background_fact("那年4月份，老板让我去参加商品交易会")
    assert is_historical_background_fact("广交会是广州重要的外贸展会")


def test_safe_filters_shared_atomic_facts_before_verification():
    checker = SAFEFactChecker(llm_adapter=None)
    mixed = [
        "2009年全球金融危机影响中国外贸出口",
        "我在天河区租了一间小房子",
    ]

    result = checker._decompose_rule_based("。".join(mixed))

    assert result == ["2009年全球金融危机影响中国外贸出口"]
