import os
from pathlib import Path

import pytest
from kwwutils import clock, printit

from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_5_1_TransformChain import (
    main,
)


@clock
@pytest.mark.testme
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    filename = "Cats&Dogs.txt"
    dirpath = os.path.dirname(os.path.abspath(__file__))
    dirpath = Path(dirpath).parent
    filename = os.path.join(dirpath, filename)
    options["filename"] = filename
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "Cats and Dogs"
    response = main(**options)
    output = response["output"].lower()
    printit(f"{model}: response", response)
    printit(f"{model}: output", output)
    assert "cat" in output
    assert "dog" in output
    assert "silvester" in output
