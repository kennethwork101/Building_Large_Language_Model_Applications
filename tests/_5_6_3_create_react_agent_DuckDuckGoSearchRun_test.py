from kwwutils import clock, printit
from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_6_3_create_react_agent_DuckDuckGoSearchRun import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "When was Avatar 2 released?"
    response = main(**options)
    printit(f"1 {model}: response", response)
    assert response["input"] == options["question"]
    assert "december 16" in response["output"].lower()
