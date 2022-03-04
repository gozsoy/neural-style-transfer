# CustomAdam definition as tf.keras.optimizers.Adam raise error when run on Mac GPUs. please see discussion and fix below.
# for nvidia gpus, ignore this and use tf.keras.optimizers.Adam with CustomAdam's hyperparameters.
# borrowed from: [MPSGraph adamUpdateWithLearningRateTensor:beta1Tensor:beta2Tensor:epsilonTensor:beta1PowerTensor:beta2PowerTensor:valuesTensor:momentumTensor:velocityTensor:gradientTensor:name:]: unrecognized selector sent to instance 0x600000eede10 - https://developer.apple.com/forums/thread/691917?answerId=701624022#701624022
#  
import tensorflow as tf


tf.config.run_functions_eagerly(True)
class CustomAdam(tf.keras.optimizers.Optimizer):
  def __init__(self, learning_rate=0.001,beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, name="CustomAdam", **kwargs):
    super().__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("beta_v", beta1)
    self._set_hyper("beta_s", beta2)
    self._set_hyper("epsilon", epsilon)
    self._set_hyper("corrected_v", beta1)
    self._set_hyper("corrected_s", beta2)
   
  def _create_slots(self, var_list):
    """
    One slot per model variable.
    """
    for var in var_list:
      self.add_slot(var, "beta_v")
      self.add_slot(var, "beta_s")
      self.add_slot(var, "epsilon")
      self.add_slot(var, "corrected_v")
      self.add_slot(var, "corrected_s")
       

  @tf.function
  def _resource_apply_dense(self, grad, var):
    """Update the slots and perform an optimization step for the model variable.
    """

    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
     
    momentum_var1 = self.get_slot(var, "beta_v")
    momentum_hyper1 = self._get_hyper("beta_v", var_dtype)
     
    momentum_var2 = self.get_slot(var, "beta_s")
    momentum_hyper2 = self._get_hyper("beta_s", var_dtype)
     
     
    momentum_var1.assign(momentum_var1 * momentum_hyper1 + (1. - momentum_hyper1)* grad)
     
    momentum_var2.assign(momentum_var2 * momentum_hyper2 + (1. - momentum_hyper2)* (grad ** 2))
     

    # Adam bias-corrected estimate
     
    corrected_v = self.get_slot(var, "corrected_v")
    corrected_v.assign(momentum_var1 / (1 - (momentum_hyper1 ** (self.iterations.numpy() + 1) )))

    corrected_s = self.get_slot(var, "corrected_s")
    corrected_s.assign(momentum_var2 / (1 - (momentum_hyper2 ** (self.iterations.numpy() + 1) )))

    epsilon_hyper = self._get_hyper("epsilon", var_dtype)
     
    var.assign_add(-lr_t * (corrected_v / (tf.sqrt(corrected_s) + epsilon_hyper)))

  def _resource_apply_sparse(self, grad, var):
    raise NotImplementedError

  def get_config(self):
    base_config = super().get_config()
    return {
      **base_config,
      "learning_rate": self._serialize_hyperparameter("learning_rate"),
      "decay": self._serialize_hyperparameter("decay"),
      "beta_v": self._serialize_hyperparameter("beta_v"),
      "beta_s": self._serialize_hyperparameter("beta_s"),
      "epsilon": self._serialize_hyperparameter("epsilon"),
    }
