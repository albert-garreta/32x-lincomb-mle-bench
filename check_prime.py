def is_prime_miller_rabin(n):
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False
    return True

p = 2**192 - 2**64 - 1
print(f"p = {p}")
print(f"bits = {p.bit_length()}")
print(f"is_prime = {is_prime_miller_rabin(p)}")

if not is_prime_miller_rabin(p):
    for delta in range(2, 1000, 2):
        c = p + delta
        if is_prime_miller_rabin(c):
            print(f"Found prime at p+{delta} = {c}")
            p = c
            break

pm1 = p - 1
for g in range(2, 200):
    if pow(g, pm1 // 2, p) != 1:
        print(f"generator = {g}")
        break
