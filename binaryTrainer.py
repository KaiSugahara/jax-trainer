import jax
import jax.numpy as jnp

from functools import partial

from baseTrainer import baseTrainer

class binaryTrainer(baseTrainer):

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        """
            損失関数
        """

        # 予測値
        pred, variables = self.model.apply({'params': params, **variables}, X, mutable=list(variables.keys()))

        # 交差エントロピー誤差
        loss = - jnp.mean(Y * jnp.log(pred) + (1 - Y) * jnp.log(1 - pred))

        return loss, variables