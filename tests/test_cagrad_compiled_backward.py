import torch
from torch._functorch import config as functorch_config

from gradientwam import runtime_factory
from gradientwam.settings import GradientWAMMethod, GradientWAMMethodConfig


def losses(weight):
    x = torch.linspace(-0.4, 0.8, 64, dtype=weight.dtype).reshape(8, 8)
    hidden = torch.sin(x @ weight)
    return hidden.square().mean(), hidden.cos().mean()


def gradients(forward):
    weight = torch.linspace(0.1, 0.9, 64, dtype=torch.float64).reshape(8, 8)
    weight.requires_grad_()
    video, action = forward(weight)
    values = []
    for loss in (video, action):
        weight.grad = None
        torch.autograd.backward(loss, inputs=(weight,), retain_graph=True)
        values.append(weight.grad.clone())
    weight.grad = None
    (video + action + 0.01 * weight.square().mean()).backward()
    values.append(weight.grad.clone())
    return values


def warm_backward(forward):
    weight = torch.linspace(0.1, 0.9, 64, dtype=torch.float64).reshape(8, 8)
    weight.requires_grad_()
    sum(forward(weight)).backward()


def test_cagrad_compiled_backwards_preserve_eager_task_and_ordinary_gradients():
    torch.set_num_threads(1)
    reference = gradients(losses)
    with functorch_config.patch(donated_buffer=True):
        runtime_factory._configure_cagrad_compilation(
            GradientWAMMethodConfig(method=GradientWAMMethod.BASELINE)
        )
        assert functorch_config.donated_buffer is True
        runtime_factory._configure_cagrad_compilation(
            GradientWAMMethodConfig(method=GradientWAMMethod.VRFM_CAGRAD)
        )
        assert functorch_config.donated_buffer is False
        compiled = torch.compile(losses, dynamic=True)
        warm_backward(compiled)
        actual = gradients(compiled)
        for value, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(value, expected, rtol=1e-10, atol=1e-12)
        print("max_abs_gradient_errors=" + str([
            float((value - expected).abs().max())
            for value, expected in zip(actual, reference, strict=True)
        ]))
        assert not torch.cuda.is_initialized()
