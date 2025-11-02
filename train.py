# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for NeRF (modernized for JAX 0.4.34 / Flax 0.7.2)."""

import functools
import gc
import time
from absl import app
from absl import flags
import threading

# ============================================================
# Global Kaggle TPU Keep-Alive (prevents ^C during JAX import)
# ============================================================
def _kaggle_keepalive():
    while True:
        print("💓 JAX import alive...", flush=True)
        time.sleep(10)

threading.Thread(target=_kaggle_keepalive, daemon=True).start()
# ============================================================

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

# ============================================================
# 🧩 FIX: JAX–Flax compatibility patch (must come before flax import)
# ============================================================
if not hasattr(jax.config, "define_bool_state"):
    def define_bool_state(name, default, help=None):
        setattr(jax.config, name, default)
    jax.config.define_bool_state = define_bool_state

import sys
try:
    from jax import linear_util as lu
except ImportError:
    import jax._src.linear_util as lu
    sys.modules["jax.linear_util"] = lu
    jax.linear_util = lu
# ============================================================

from torch.utils import tensorboard
from flax.training import checkpoints, train_state
from flax.jax_utils import replicate, prefetch_to_device
import optax

from internal import datasets
from internal import math
from internal import models
from internal import utils
from internal import vis

FLAGS = flags.FLAGS
utils.define_common_flags()
flags.DEFINE_integer('render_every', 5000,
                     'The number of steps between test set image renderings.')

jax.config.parse_flags_with_absl()


def train_step(model, config, rng, state, batch, lr):
    """One optimization step."""
    rng, key = random.split(rng)

    def loss_fn(params):
        def tree_sum_fn(fn):
            return jax.tree_util.tree_reduce(
                lambda x, y: x + fn(y), params, initializer=0
            )

        weight_l2 = config.weight_decay_mult * (
            tree_sum_fn(lambda z: jnp.sum(z**2)) /
            tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape)))
        )

        ret = model.apply(
            params,
            key,
            batch['rays'],
            randomized=config.randomized,
            white_bkgd=config.white_bkgd
        )

        mask = batch['rays'].lossmult
        if config.disable_multiscale_loss:
            mask = jnp.ones_like(mask)

        losses = []
        for (rgb, _, _) in ret:
            losses.append(
                (mask * (rgb - batch['pixels'][..., :3])**2).sum() / mask.sum()
            )
        losses = jnp.array(losses)

        loss = (
            config.coarse_loss_mult * jnp.sum(losses[:-1]) + losses[-1] + weight_l2
        )

        stats = utils.Stats(
            loss=loss,
            losses=losses,
            weight_l2=weight_l2,
            psnr=0.0,
            psnrs=0.0,
            grad_norm=0.0,
            grad_abs_max=0.0,
            grad_norm_clipped=0.0,
        )
        return loss, stats

    (_, stats), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grad = jax.lax.pmean(grad, axis_name='batch')
    stats = jax.lax.pmean(stats, axis_name='batch')

    def tree_norm(tree):
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + jnp.sum(y**2), tree, initializer=0
            )
        )

    if config.grad_max_val > 0:
        grad = jax.tree_util.tree_map(
            lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), grad
        )

    grad_abs_max = jax.tree_util.tree_reduce(
        lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), grad, initializer=0
    )

    grad_norm = tree_norm(grad)
    if config.grad_max_norm > 0:
        mult = jnp.minimum(1, config.grad_max_norm / (1e-7 + grad_norm))
        grad = jax.tree_util.tree_map(lambda z: mult * z, grad)
    grad_norm_clipped = tree_norm(grad)

    new_state = state.apply_gradients(grads=grad)
    psnrs = math.mse_to_psnr(stats.losses)
    stats = utils.Stats(
        loss=stats.loss,
        losses=stats.losses,
        weight_l2=stats.weight_l2,
        psnr=psnrs[-1],
        psnrs=psnrs,
        grad_norm=grad_norm,
        grad_abs_max=grad_abs_max,
        grad_norm_clipped=grad_norm_clipped,
    )

    return new_state, stats, rng

def main(unused_argv):
    rng = random.PRNGKey(20200823)

    # ============================================================
    # ✅ Kaggle TPU Keep-Alive Thread
    # Prevents unexpected "^C" interrupts caused by TPU watchdog.
    # ============================================================
    def keepalive():
        while True:
            print("💓 TPU alive...", flush=True)
            time.sleep(20)

    threading.Thread(target=keepalive, daemon=True).start()
    # ============================================================

    np.random.seed(20201473 + jax.host_id())
    config = utils.load_config()


    if config.batch_size % jax.device_count() != 0:
        raise ValueError('Batch size must be divisible by the number of devices.')

    dataset = datasets.get_dataset('train', FLAGS.data_dir, config)
    test_dataset = datasets.get_dataset('test', FLAGS.data_dir, config)

    rng, key = random.split(rng)
    model, variables = models.construct_mipnerf(key, dataset.peek())
    num_params = jax.tree_util.tree_reduce(
        lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0
    )
    print(f'Number of parameters being optimized: {num_params}')

    # Optax + TrainState
    tx = optax.adam(config.lr_init)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables, tx=tx)
    del variables

    learning_rate_fn = functools.partial(
        math.learning_rate_decay,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        max_steps=config.max_steps,
        lr_delay_steps=config.lr_delay_steps,
        lr_delay_mult=config.lr_delay_mult,
    )

    train_pstep = jax.pmap(
        functools.partial(train_step, model, config),
        axis_name='batch',
        in_axes=(0, 0, 0, None),
        donate_argnums=(2,),
    )

    # Evaluation fn
    def render_eval_fn(params, _, rays):
        return jax.lax.all_gather(
            model.apply(
                params,
                random.PRNGKey(0),
                rays,
                randomized=False,
                white_bkgd=config.white_bkgd,
            ),
            axis_name='batch',
        )

    render_eval_pfn = jax.pmap(
        render_eval_fn, in_axes=(None, None, 0), donate_argnums=(2,), axis_name='batch'
    )

    ssim_fn = jax.jit(functools.partial(math.compute_ssim, max_val=1.))

    if not utils.isdir(FLAGS.train_dir):
        utils.makedirs(FLAGS.train_dir)

    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    init_step = int(state.step)
    state = replicate(state)

    if jax.host_id() == 0:
        summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)

    pdataset = prefetch_to_device(dataset, 3)
    rng = rng + jax.host_id()
    keys = random.split(rng, jax.local_device_count())
    gc.disable()

    stats_trace = []
    reset_timer = True
    for step, batch in zip(range(init_step, config.max_steps + 1), pdataset):
        if reset_timer:
            t_loop_start = time.time()
            reset_timer = False

        lr = learning_rate_fn(step)
        state, stats, keys = train_pstep(keys, state, batch, lr)
        if jax.host_id() == 0:
            stats_trace.append(stats)
        if step % config.gc_every == 0:
            gc.collect()

        if jax.host_id() == 0:
            if step % config.print_every == 0:
                summary_writer.scalar('num_params', num_params, step)
                summary_writer.scalar('train_loss', stats.loss[0], step)
                summary_writer.scalar('train_psnr', stats.psnr[0], step)
                for i, l in enumerate(stats.losses[0]):
                    summary_writer.scalar(f'train_losses_{i}', l, step)
                for i, p in enumerate(stats.psnrs[0]):
                    summary_writer.scalar(f'train_psnrs_{i}', p, step)
                summary_writer.scalar('weight_l2', stats.weight_l2[0], step)
                avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
                avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
                max_grad_norm = np.max(np.concatenate([s.grad_norm for s in stats_trace]))
                avg_grad_norm = np.mean(np.concatenate([s.grad_norm for s in stats_trace]))
                max_clipped_grad_norm = np.max(
                    np.concatenate([s.grad_norm_clipped for s in stats_trace])
                )
                max_grad_max = np.max(np.concatenate([s.grad_abs_max for s in stats_trace]))
                stats_trace = []
                summary_writer.scalar('train_avg_loss', avg_loss, step)
                summary_writer.scalar('train_avg_psnr', avg_psnr, step)
                summary_writer.scalar('train_max_grad_norm', max_grad_norm, step)
                summary_writer.scalar('train_avg_grad_norm', avg_grad_norm, step)
                summary_writer.scalar('train_max_clipped_grad_norm',
                                      max_clipped_grad_norm, step)
                summary_writer.scalar('train_max_grad_max', max_grad_max, step)
                summary_writer.scalar('learning_rate', lr, step)
                steps_per_sec = config.print_every / (time.time() - t_loop_start)
                reset_timer = True
                rays_per_sec = config.batch_size * steps_per_sec
                summary_writer.scalar('train_steps_per_sec', steps_per_sec, step)
                summary_writer.scalar('train_rays_per_sec', rays_per_sec, step)
                precision = int(np.ceil(np.log10(config.max_steps))) + 1
                print(('{:' + '{:d}'.format(precision) + 'd}').format(step) +
                      f'/{config.max_steps:d}: ' + f'i_loss={stats.loss[0]:0.4f}, ' +
                      f'avg_loss={avg_loss:0.4f}, ' +
                      f'weight_l2={stats.weight_l2[0]:0.2e}, ' + f'lr={lr:0.2e}, ' +
                      f'{rays_per_sec:0.0f} rays/sec')
            if step % config.save_every == 0:
                state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
                checkpoints.save_checkpoint(
                    FLAGS.train_dir, state_to_save, int(step), keep=100
                )

        if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
            t_eval_start = time.time()
            eval_params = jax.device_get(jax.tree_map(lambda x: x[0], state)).params
            test_case = next(test_dataset)
            pred_color, pred_distance, pred_acc = models.render_image(
                functools.partial(render_eval_pfn, eval_params),
                test_case['rays'],
                keys[0],
                chunk=FLAGS.chunk,
            )
            vis_suite = vis.visualize_suite(pred_distance, pred_acc)

            if jax.host_id() == 0:
                psnr = math.mse_to_psnr(((pred_color - test_case['pixels'])**2).mean())
                ssim = ssim_fn(pred_color, test_case['pixels'])
                eval_time = time.time() - t_eval_start
                num_rays = jnp.prod(jnp.array(test_case['rays'].directions.shape[:-1]))
                rays_per_sec = num_rays / eval_time
                summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
                print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')
                summary_writer.scalar('test_psnr', psnr, step)
                summary_writer.scalar('test_ssim', ssim, step)
                summary_writer.image('test_pred_color', pred_color, step)
                for k, v in vis_suite.items():
                    summary_writer.image('test_pred_' + k, v, step)
                summary_writer.image('test_pred_acc', pred_acc, step)
                summary_writer.image('test_target', test_case['pixels'], step)

    if config.max_steps % config.save_every != 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.train_dir, state, int(config.max_steps), keep=100
        )


if __name__ == '__main__':
    app.run(main)
