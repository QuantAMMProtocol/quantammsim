"""
Integration tests for Adam optimizer in the training pipeline.

These tests verify that the Adam optimizer integrates correctly with
the historic data training workflow.
"""
import pytest
import copy

from quantammsim.runners.jax_runners import train_on_historic_data
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set, check_run_fingerprint
from tests.conftest import TEST_DATA_DIR


@pytest.fixture
def adam_training_fingerprint(training_fingerprint):
    """Create a fingerprint configured for Adam optimizer testing."""
    fp = copy.deepcopy(training_fingerprint)
    fp["optimisation_settings"]["optimiser"] = "adam"
    fp["optimisation_settings"]["base_lr"] = 0.01
    fp["optimisation_settings"]["n_iterations"] = 3
    fp["optimisation_settings"]["n_cycles"] = 1
    fp["optimisation_settings"]["n_parameter_sets"] = 1
    fp["optimisation_settings"]["batch_size"] = 2
    return fp


@pytest.fixture
def sgd_training_fingerprint(training_fingerprint):
    """Create a fingerprint configured for SGD optimizer testing."""
    fp = copy.deepcopy(training_fingerprint)
    fp["optimisation_settings"]["optimiser"] = "sgd"
    fp["optimisation_settings"]["base_lr"] = 0.01
    fp["optimisation_settings"]["n_iterations"] = 3
    fp["optimisation_settings"]["n_cycles"] = 1
    fp["optimisation_settings"]["n_parameter_sets"] = 1
    fp["optimisation_settings"]["batch_size"] = 2
    return fp


class TestAdamOptimizer:
    """Test suite for Adam optimizer integration."""

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    def test_adam_training_completes(self, adam_training_fingerprint):
        """Test that training with Adam optimizer completes without errors."""
        # Prepare fingerprint
        recursive_default_set(adam_training_fingerprint, run_fingerprint_defaults)
        check_run_fingerprint(adam_training_fingerprint)

        # Run training
        result = train_on_historic_data(
            adam_training_fingerprint,
            iterations_per_print=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        # Verify result is returned
        assert result is not None

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    def test_sgd_training_completes(self, sgd_training_fingerprint):
        """Test that training with SGD optimizer completes without errors."""
        # Prepare fingerprint
        recursive_default_set(sgd_training_fingerprint, run_fingerprint_defaults)
        check_run_fingerprint(sgd_training_fingerprint)

        # Run training
        result = train_on_historic_data(
            sgd_training_fingerprint,
            iterations_per_print=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        # Verify result is returned
        assert result is not None

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    def test_adam_optimizer_selection(self, adam_training_fingerprint):
        """Verify Adam is correctly selected as the optimizer."""
        recursive_default_set(adam_training_fingerprint, run_fingerprint_defaults)

        assert adam_training_fingerprint["optimisation_settings"]["optimiser"] == "adam"

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    def test_adam_with_different_learning_rates(self, adam_training_fingerprint):
        """Test Adam with various learning rates."""
        learning_rates = [0.001, 0.01, 0.1]

        for lr in learning_rates:
            fp = copy.deepcopy(adam_training_fingerprint)
            fp["optimisation_settings"]["base_lr"] = lr
            fp["optimisation_settings"]["n_iterations"] = 2
            fp["optimisation_settings"]["n_cycles"] = 1

            recursive_default_set(fp, run_fingerprint_defaults)
            check_run_fingerprint(fp)

            result = train_on_historic_data(
                fp,
                iterations_per_print=1,
                verbose=False,
                root=TEST_DATA_DIR,
            )

            assert result is not None, f"Training failed with learning rate {lr}"
