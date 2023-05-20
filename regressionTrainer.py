import jax
import jax.numpy as jnp

from functools import partial

from baseTrainer import baseTrainer

class regressionTrainer(baseTrainer):

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        """
            損失関数
        """

        # 予測値
        pred, variables = self.model.apply({'params': params, **variables}, X, mutable=list(variables.keys()))

        # MSEを計算
        loss = jnp.mean((pred - Y)**2)

        return loss, variables