import builtins, inspect, traceback
import jax.numpy as jnp

orig_sum = builtins.sum
def debug_sum(x, *args, **kwargs):
    print(f">>> builtins.sum on type: {type(x)}")
    # show a short stack to locate who called sum
    for line in traceback.format_stack(limit=4)[:-1]:
        print("    ", line.strip())
    return orig_sum(x, *args, **kwargs)

builtins.sum = debug_sum

a = jnp.arange(5)
print("Calling Python sum(a):")
_ = sum(a)

print("\nCalling jnp.sum(a):")
_ = jnp.sum(a)

builtins.sum = orig_sum
