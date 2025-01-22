from kwwutils import clock, printit
from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_3_SimpleSequentialChain import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "Cats and Dogs"
    response = main(**options)
    printit(f"{model}: response", response)
    assert response["input"] == options["question"]
