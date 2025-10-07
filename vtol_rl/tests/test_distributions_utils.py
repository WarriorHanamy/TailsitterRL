import torch

from vtol_rl.utils.type import Uniform, Gaussian


def test_uniform_basics():
    t = Uniform.from_min_max(1.0, 3.0)
    mm = t.sample(size=1000)
    print(torch.min(mm.view(1, -1)))
    print(torch.max(mm.view(1, -1)))

    samples = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0, 3.0]).view(-1, 2)
    print(f"samples.shape: {samples.shape},\n {samples}")
    normed_samples = t.per_sample_normalize(samples)
    print(f"normed_samples.shape: {normed_samples.shape},\n {normed_samples}")
    print(t.radius)
    print(t.mean)

    print(normed_samples.abs().view(1, -1))
    assert torch.all(normed_samples.abs().view(1, -1) <= 1.0)

    print(f"samples: {samples.view(1, -1)}")
    print(f"normed_samples: {normed_samples.view(1, -1)}")


def test_uniform_initialization():
    # TEST float
    single_mean = 4.0
    single_radius = 2.0
    t = Uniform(single_mean, single_radius)
    assert t.mean.shape == (1, 1)

    # TEST list
    single_mean = [4.0]
    single_radius = [2.0]
    t = Uniform(single_mean, single_radius)
    assert t.mean.shape == (1, 1)

    # TEST torch.Tensor
    mean = torch.randn(4)
    radius = torch.tensor(4)
    t = Uniform(mean, radius)
    assert t.mean.shape == (1, 4)

    mean = mean.view(-1, 1)
    radius = radius.view(-1, 1)
    t = Uniform(mean, radius)
    assert t.mean.shape == (1, 4)

    mean = mean.view(1, -1)
    radius = radius.view(1, -1)
    t = Uniform(mean, radius)
    assert t.mean.shape == (1, 4)

    # TEST dict
    position = {"mean": [0.0, 0.0, 0.0], "radius": [1.0, 2.0, 3.0]}
    orientation = {"mean": [0.0, 0.0, 0.0], "radius": [0.0, 0.0, 0.0]}  # euler angle
    velocity = {"mean": [0.0, 0.0, 0.0], "radius": [0.0, 0.0, 0.0]}
    angular_velocity = {"mean": [0.0, 0.0, 0.0], "radius": [0.0, 0.0, 0.0]}

    position_sampler = Uniform(**position)
    orientation_sampler = Uniform(**orientation)
    velocity_sampler = Uniform(**velocity)
    angular_velocity_sampler = Uniform(**angular_velocity)

    assert orientation_sampler.mean.shape == (1, 3)
    assert velocity_sampler.mean.shape == (1, 3)
    assert angular_velocity_sampler.mean.shape == (1, 3)

    assert position_sampler.mean.shape == (1, 3)
    assert position_sampler.sample(size=10).shape == (10, 3)
    print(position_sampler.sample(size=10))


def test_uniform_functionalities():
    # TEST denormalization
    position = {"mean": [0.0, 0.0, 0.0], "radius": [1.0, 2.0, 3.0]}
    position_sampler = Uniform(**position)
    num_samples = 10
    samples = position_sampler.sample(size=num_samples)

    normed_samples = position_sampler.per_sample_normalize(samples)
    assert normed_samples.shape == (num_samples, 3)
    print(f"normed_samples:\n {normed_samples}")

    in_range = (normed_samples >= -1) & (normed_samples <= 1)
    print(in_range)
    all_conditions_features = torch.all(in_range, dim=1)
    assert all_conditions_features.shape == (num_samples,)

    all_conditions = torch.all(all_conditions_features, dim=0)
    assert all_conditions.shape == ()

    assert all_conditions.all()  #

    print(all_conditions_features)
    print(f"normed_samples.shape: {normed_samples.shape}")


def test_gaussian_initialization():
    # TEST float
    single_mean = 4.0
    single_std = 2.0
    t = Gaussian(single_mean, single_std)
    assert t.mean.shape == (1, 1)
    assert t.std.shape == (1, 1)

    # TEST list
    single_mean = [4.0]
    single_std = [2.0]
    t = Gaussian(single_mean, single_std)
    assert t.mean.shape == (1, 1)
    assert t.std.shape == (1, 1)

    # TEST torch.Tensor
    mean = torch.randn(4)
    std = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t = Gaussian(mean, std)
    assert t.mean.shape == (1, 4)
    assert t.std.shape == (1, 4)

    mean = mean.view(-1, 1)
    std = std.view(-1, 1)
    t = Gaussian(mean, std)
    assert t.mean.shape == (1, 4)
    assert t.std.shape == (1, 4)

    mean = mean.view(1, -1)
    std = std.view(1, -1)
    t = Gaussian(mean, std)
    assert t.mean.shape == (1, 4)
    assert t.std.shape == (1, 4)

    # TEST dict
    position = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 2.0, 3.0]}
    orientation = {"mean": [0.0, 0.0, 0.0], "std": [0.1, 0.1, 0.1]}
    velocity = {"mean": [0.0, 0.0, 0.0], "std": [0.5, 0.5, 0.5]}
    angular_velocity = {"mean": [0.0, 0.0, 0.0], "std": [0.2, 0.2, 0.2]}

    position_sampler = Gaussian(**position)
    orientation_sampler = Gaussian(**orientation)
    velocity_sampler = Gaussian(**velocity)
    angular_velocity_sampler = Gaussian(**angular_velocity)

    assert orientation_sampler.mean.shape == (1, 3)
    assert velocity_sampler.mean.shape == (1, 3)
    assert angular_velocity_sampler.mean.shape == (1, 3)

    assert position_sampler.mean.shape == (1, 3)
    assert position_sampler.sample(size=10).shape == (10, 3)
    print(position_sampler.sample(size=10))


def test_gaussian_from_min_max():
    # Test single value
    min_val = 0.0
    max_val = 6.0
    g = Gaussian.from_min_max(min_val, max_val)
    assert torch.allclose(g.mean, torch.tensor([[3.0]]))
    assert torch.allclose(g.std, torch.tensor([[1.0]]))  # (6-0)/6 = 1

    # Test multiple values
    min_vals = [0.0, -6.0, 10.0]
    max_vals = [6.0, 6.0, 20.0]
    g = Gaussian.from_min_max(min_vals, max_vals)
    expected_mean = torch.tensor([[3.0, 0.0, 15.0]])
    expected_std = torch.tensor([[1.0, 2.0, 10.0 / 6]])
    assert torch.allclose(g.mean, expected_mean)
    assert torch.allclose(g.std, expected_std)


def test_gaussian_functionalities():
    # Test normalization and denormalization
    position = {"mean": [1.0, 2.0, 3.0], "std": [0.5, 1.0, 1.5]}
    position_sampler = Gaussian(**position)
    num_samples = 10
    samples = position_sampler.sample(size=num_samples)

    # Test normalization
    normed_samples = position_sampler.per_sample_normalize(samples)
    assert normed_samples.shape == (num_samples, 3)
    print(f"normed_samples:\n {normed_samples}")

    # Check if normalized samples have mean ~0 and std ~1
    sample_mean = normed_samples.mean(dim=0)
    sample_std = normed_samples.std(dim=0)

    # Test denormalization
    denormed_samples = position_sampler.per_sample_denormalize(normed_samples)
    assert denormed_samples.shape == (num_samples, 3)
    assert torch.allclose(denormed_samples, samples, atol=1e-6)
