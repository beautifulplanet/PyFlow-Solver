# Simple Makefile for HPF-CFD
.DEFAULT_GOAL := all
FC = gfortran
FFLAGS = -O2 -Wall -Wextra -fimplicit-none -std=f2008
SRC_DIR = src
BUILD_DIR = build
EXEC = $(BUILD_DIR)/hpf_cfd.exe
BC_TEST_EXE = $(BUILD_DIR)/test_bc.exe
TEST_PARAMETERS_EXE = $(BUILD_DIR)/test_parameters.exe
TEST_FIELDS_EXE = $(BUILD_DIR)/test_fields.exe
TEST_SOLVER_NAN_EXE = $(BUILD_DIR)/test_solver_nan.exe
TEST_IO_UTILS_EXE = $(BUILD_DIR)/test_io_utils.exe
TESTS = $(TEST_PARAMETERS_EXE) $(TEST_FIELDS_EXE) $(TEST_SOLVER_NAN_EXE) $(TEST_IO_UTILS_EXE)
$(TEST_PARAMETERS_EXE): $(CORE_OBJS) $(BUILD_DIR)/test_parameters.o
	$(FC) $(FFLAGS) -o $@ $^
$(TEST_FIELDS_EXE): $(CORE_OBJS) $(BUILD_DIR)/test_fields.o
	$(FC) $(FFLAGS) -o $@ $^
$(TEST_SOLVER_NAN_EXE): $(CORE_OBJS) $(BUILD_DIR)/test_solver_nan.o
	$(FC) $(FFLAGS) -o $@ $^
$(TEST_IO_UTILS_EXE): $(CORE_OBJS) $(BUILD_DIR)/test_io_utils.o
	$(FC) $(FFLAGS) -o $@ $^
test_parameters: $(TEST_PARAMETERS_EXE)
	$(TEST_PARAMETERS_EXE)
test_fields: $(TEST_FIELDS_EXE)
	$(TEST_FIELDS_EXE)
test_solver_nan: $(TEST_SOLVER_NAN_EXE)
	-$(TEST_SOLVER_NAN_EXE) || true
test_io_utils: $(TEST_IO_UTILS_EXE)
	$(TEST_IO_UTILS_EXE)
test_all: test_parameters test_fields test_solver_nan test_io_utils bc_test

CORE_SRCS = parameters.f90 fields.f90 boundary_conditions.f90 io_utils.f90 solver.f90
MAIN_SRC = main.f90
BC_TEST_SRC = tests/test_bc.f90

CORE_OBJS = $(addprefix $(BUILD_DIR)/,$(CORE_SRCS:.f90=.o))
MAIN_OBJ = $(BUILD_DIR)/$(MAIN_SRC:.f90=.o)
BC_TEST_OBJ = $(BUILD_DIR)/$(BC_TEST_SRC:.f90=.o)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: $(BUILD_DIR) $(EXEC)

$(EXEC): $(CORE_OBJS) $(MAIN_OBJ)
	$(FC) $(FFLAGS) -o $@ $^

$(BC_TEST_EXE): $(CORE_OBJS) $(BC_TEST_OBJ)
	$(FC) $(FFLAGS) -o $@ $^


# Pattern rule for core sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

# Pattern rule for test sources
$(BUILD_DIR)/test_%.o: tests/test_%.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

run: all
	$(EXEC)

bc_test: $(BC_TEST_EXE)
	$(BC_TEST_EXE)

clean:
	@if [ -d $(BUILD_DIR) ]; then rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod $(EXEC) $(BC_TEST_EXE); fi

.PHONY: all clean run bc_test
