import pytest
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from omegaconf import OmegaConf
from collections import Counter

from visreps import utils
from visreps.models.custom_cnn import TinyCustomCNN # For testing param groups

# --- Fixtures ---

@pytest.fixture
def simple_model():
    """A simple model with parameters, including some without decay (bias, norm)."""
    return nn.Sequential(
        nn.Linear(10, 5, bias=True),
        nn.BatchNorm1d(5),
        nn.ReLU(),
        nn.Linear(5, 2, bias=False) # No bias on last layer
    )

@pytest.fixture
def tiny_cnn_model():
     """A TinyCustomCNN model instance for more complex param group testing."""
     default_trainable = {'conv': '11111', 'fc': '111'}
     return TinyCustomCNN(num_classes=10, trainable_layers=default_trainable)

@pytest.fixture
def base_util_cfg():
    """Base configuration for optimizer and scheduler setup tests."""
    return OmegaConf.create({
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "lr_scheduler": "StepLR",
        # Specific scheduler params moved into scheduler_kwargs
        "scheduler_kwargs": {
            "step_size": 5,    # For StepLR
            "gamma": 0.1,      # For StepLR, MultiStepLR
            "milestones": [8, 15], # For MultiStepLR
            # T_max for CosineAnnealingLR will be derived from num_epochs
        },
        "warmup_epochs": 0,
        "num_epochs": 20, # Needed for CosineAnnealingLR
        # Add other necessary keys if utils functions depend on them
    })

# --- Optimizer Setup Tests (using simple_model) ---

@pytest.mark.parametrize(
    "optimizer_name, expected_type, lr, wd",
    [
        ("Adam", optim.Adam, 0.01, 0.0005),
        ("SGD", optim.SGD, 0.001, 0.0),
        ("AdamW", optim.AdamW, 0.05, 0.01),
    ]
)
def test_setup_optimizer_type_lr_wd(simple_model, base_util_cfg, optimizer_name, expected_type, lr, wd):
    """Test: Verify setup_optimizer creates the specified optimizer type with correct LR and WD."""
    base_util_cfg.optimizer = optimizer_name
    base_util_cfg.learning_rate = lr
    base_util_cfg.weight_decay = wd

    optimizer = utils.setup_optimizer(simple_model, base_util_cfg)

    assert isinstance(optimizer, expected_type)
    assert optimizer.defaults['lr'] == lr
    # Weight decay is handled per group, check group settings below

def test_setup_optimizer_param_groups(simple_model, base_util_cfg):
    """Test: Verify setup_optimizer creates correct parameter groups (decay vs no_decay)."""
    cfg = base_util_cfg.copy()
    cfg.optimizer = "AdamW" # Use AdamW as example
    cfg.weight_decay = 0.05

    optimizer = utils.setup_optimizer(simple_model, cfg)

    assert len(optimizer.param_groups) == 2
    group0_params = optimizer.param_groups[0]['params']
    group1_params = optimizer.param_groups[1]['params']
    group0_wd = optimizer.param_groups[0]['weight_decay']
    group1_wd = optimizer.param_groups[1]['weight_decay']

    # Identify which group is decay and which is no_decay
    decay_group_params = group0_params if group0_wd > 0 else group1_params
    no_decay_group_params = group1_params if group0_wd > 0 else group0_params
    decay_group_wd = max(group0_wd, group1_wd)
    no_decay_group_wd = min(group0_wd, group1_wd)

    assert decay_group_wd == cfg.weight_decay
    assert no_decay_group_wd == 0.0

    # Check parameters are correctly assigned
    decay_param_names = {p_name for p_name, p in simple_model.named_parameters() if id(p) in [id(p_opt) for p_opt in decay_group_params]}
    no_decay_param_names = {p_name for p_name, p in simple_model.named_parameters() if id(p) in [id(p_opt) for p_opt in no_decay_group_params]}

    # Expected no_decay: bias terms, normalization layer weights/biases
    expected_no_decay = {'0.bias', '1.weight', '1.bias'} # linear1 bias, bn weight, bn bias
    expected_decay = {'0.weight', '3.weight'} # linear1 weight, linear2 weight (no bias)

    assert decay_param_names == expected_decay
    assert no_decay_param_names == expected_no_decay

def test_setup_optimizer_invalid(simple_model, base_util_cfg):
    """Test: Ensure invalid optimizer name raises ValueError."""
    cfg = base_util_cfg.copy()
    cfg.optimizer = "InvalidOptimizer"
    with pytest.raises(ValueError, match="Unknown optimizer: InvalidOptimizer"):
        utils.setup_optimizer(simple_model, cfg)

# --- Scheduler Setup Tests (using tiny_cnn_model for optimizer) ---

@pytest.fixture
def optimizer_for_scheduler(tiny_cnn_model, base_util_cfg):
     """Provides an optimizer instance needed for scheduler setup."""
     return utils.setup_optimizer(tiny_cnn_model, base_util_cfg)

@pytest.mark.parametrize("scheduler_name, expected_type", [
    ("StepLR", lr_scheduler.StepLR),
    ("MultiStepLR", lr_scheduler.MultiStepLR),
    ("CosineAnnealingLR", lr_scheduler.CosineAnnealingLR),
])
def test_setup_scheduler_no_warmup(optimizer_for_scheduler, base_util_cfg, scheduler_name, expected_type):
    """Test: Verify scheduler creation without warmup."""
    cfg = base_util_cfg.copy()
    cfg.lr_scheduler = scheduler_name
    cfg.warmup_epochs = 0 # Ensure no warmup

    # Create a deep copy of scheduler_kwargs to avoid modification issues
    if hasattr(cfg, 'scheduler_kwargs'):
        cfg.scheduler_kwargs = OmegaConf.to_container(cfg.scheduler_kwargs, resolve=True)
    else:
        cfg.scheduler_kwargs = {}

    scheduler = utils.setup_scheduler(optimizer_for_scheduler, cfg)
    assert isinstance(scheduler, expected_type)

    # Check specific parameters against cfg.scheduler_kwargs
    if scheduler_name == "StepLR":
        assert scheduler.step_size == cfg.scheduler_kwargs["step_size"]
        assert scheduler.gamma == cfg.scheduler_kwargs["gamma"]
    elif scheduler_name == "MultiStepLR":
        assert hasattr(scheduler, "milestones"), "MultiStepLR scheduler should have 'milestones' attribute"
        assert scheduler.milestones == Counter(cfg.scheduler_kwargs["milestones"])
        assert scheduler.gamma == cfg.scheduler_kwargs["gamma"]
    elif scheduler_name == "CosineAnnealingLR":
        # T_max is num_epochs - warmup_epochs (0 here)
        assert scheduler.T_max == cfg.num_epochs

@pytest.mark.parametrize("scheduler_name", ["StepLR", "MultiStepLR", "CosineAnnealingLR"])
def test_setup_scheduler_with_warmup(optimizer_for_scheduler, base_util_cfg, scheduler_name):
    """Test: Verify scheduler creation with warmup (SequentialLR)."""
    cfg = base_util_cfg.copy()
    cfg.lr_scheduler = scheduler_name
    cfg.warmup_epochs = 3 # Enable warmup

    # Create a deep copy of scheduler_kwargs
    if hasattr(cfg, 'scheduler_kwargs'):
        cfg.scheduler_kwargs = OmegaConf.to_container(cfg.scheduler_kwargs, resolve=True)
    else:
        cfg.scheduler_kwargs = {}

    scheduler = utils.setup_scheduler(optimizer_for_scheduler, cfg)
    assert isinstance(scheduler, lr_scheduler.SequentialLR)
    assert len(scheduler._schedulers) == 2
    assert isinstance(scheduler._schedulers[0], lr_scheduler.LinearLR) # Warmup scheduler
    assert scheduler._milestones == [cfg.warmup_epochs]

    # Check warmup scheduler parameters
    warmup_sched = scheduler._schedulers[0]
    assert warmup_sched.total_iters == cfg.warmup_epochs

    # Check main scheduler parameters (created using cfg.scheduler_kwargs)
    main_sched = scheduler._schedulers[1]
    if scheduler_name == "StepLR":
        assert isinstance(main_sched, lr_scheduler.StepLR)
        assert main_sched.step_size == cfg.scheduler_kwargs["step_size"]
        assert main_sched.gamma == cfg.scheduler_kwargs["gamma"]
    elif scheduler_name == "MultiStepLR":
        assert isinstance(main_sched, lr_scheduler.MultiStepLR)
        assert hasattr(main_sched, "milestones"), "MultiStepLR main scheduler should have 'milestones' attribute"
        assert main_sched.milestones == Counter(cfg.scheduler_kwargs["milestones"])
        assert main_sched.gamma == cfg.scheduler_kwargs["gamma"]
    elif scheduler_name == "CosineAnnealingLR":
        assert isinstance(main_sched, lr_scheduler.CosineAnnealingLR)
        assert main_sched.T_max == cfg.num_epochs - cfg.warmup_epochs

def test_setup_scheduler_invalid(optimizer_for_scheduler, base_util_cfg):
    """Test: Ensure invalid scheduler name raises ValueError."""
    cfg = base_util_cfg.copy()
    cfg.lr_scheduler = "InvalidScheduler"
    with pytest.raises(ValueError, match="Invalid LR scheduler name: InvalidScheduler"):
        utils.setup_scheduler(optimizer_for_scheduler, cfg) 