from kwwutils import clock, printit
from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_6_1_initialize_agent_SerpAPIWrapper import (
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
    printit(f"2 {model}: response type", type(response))
    printit(f"3 {model}: response keys", response.keys())
    output = response["output"].lower()
    printit(f"4 {model}: output", output)
    printit(f"5 {model}: question", options["question"])
    printit(f"6 {model}: input", response["input"])
    assert response["input"] == options["question"]
    assert "avatar 2" in output
