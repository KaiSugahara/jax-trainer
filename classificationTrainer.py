import jax
import jax.numpy as jnp

from functools import partial

from baseTrainer import baseTrainer

class classificationTrainer(baseTrainer):

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        """
            損失関数
        """

        # 予測値
        pred, variables = self.model.apply({'params': params, **variables}, X, mutable=list(variables.keys()))

        # 交差エントロピー誤差
        loss = - jnp.mean(jnp.sum(Y * jnp.log(pred), axis=1))

        return loss, variables