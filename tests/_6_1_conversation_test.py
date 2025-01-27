from kwwutils import clock, printit

from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._6_1_conversation import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    response = main(**options)
    printit(f"1 {model}: response", response)
    printit(f"2 {model}: response.keys()", sorted(response.keys()))
    question = "What kind of other events?"
    assert sorted(response.keys()) == ["chat_history", "question", "text"]
    assert response["question"] == question
    assert "japan" in response["text"].lower()
