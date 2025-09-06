#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"
#include "../include/CFD_Engine.h"


TEST_CASE("OpenFOAM solver increments all field values by 1.0", "[OpenFOAMSolver]") {
    CFD_Engine engine("openfoam", 5);
    // Field should be initialized to 0.0
    for (size_t i = 0; i < engine.field().size(); ++i) {
        REQUIRE(engine.field().get(i) == 0.0);
    }
    engine.run();
    for (size_t i = 0; i < engine.field().size(); ++i) {
        REQUIRE(engine.field().get(i) == 1.0);
    }
}

TEST_CASE("Experimental solver sets field values to their indices", "[ExperimentalSolver]") {
    CFD_Engine engine("experimental", 5);
    // Field should be initialized to 0.0
    for (size_t i = 0; i < engine.field().size(); ++i) {
        REQUIRE(engine.field().get(i) == 0.0);
    }
    engine.run();
    for (size_t i = 0; i < engine.field().size(); ++i) {
        REQUIRE(engine.field().get(i) == static_cast<double>(i));
    }
}
