extern crate rand;

use rand::prelude::*;
use rand::distributions::{Uniform, Exp1};
use std::collections::{HashSet, HashMap, BTreeSet};
use std::time::{Instant, Duration};

fn main() {
    fn do_test<L: FnMut(usize, usize) -> SampleResult>(mut lambda: L, name: &str, mut max_amount: usize, max_length: usize) {
        if max_amount == 0 {
            max_amount = 1_000_000;
        }
        for length in vec![10, 100, 500, 1000, 1_000_000, 1_000_000_000/**/] {
            if max_length != 0 && length > max_length {
                continue;
            }
            let flen = std::cmp::min(1_000_000, length) as f32;
            let mut amounts = 
                [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                 (0.01 * flen) as usize,
                 (0.02 * flen) as usize,
                 (0.05 * flen) as usize,
                 (0.1 * flen) as usize,
                 (0.2 * flen) as usize,
                 (0.5 * flen) as usize,
                 (0.9 * flen) as usize,
                 (0.95 * flen) as usize]
                .iter().cloned()
                .filter(|&x| 0 < x && x <= length && x < max_amount)
                .collect::<Vec<usize>>();
            amounts.sort();
            amounts.dedup();
            for amount in amounts {
                let bench_start = Instant::now();
                let mut samples = Vec::new();
                loop {
                    let start = Instant::now();
                    let res = lambda(length, amount);
                    verify(res, length, amount);
                    let dur = start.elapsed();
                    samples.push(nanos(&dur));
                    let time = nanos(&bench_start.elapsed());
                    if (samples.len() >= 10 && time > 1_000_000_000) || time > 10_000_000_000 {
                        break;
                    }
                }
                let time = samples.iter().sum::<u64>() as f32 / (samples.len() * amount) as f32;
                println!("{: >10}, {: >7}, {: >10.2}, {: >33}, {}", length, amount, time, name, bench_start.elapsed().as_secs());
            }
        }
    }

    let r = &mut StdRng::from_entropy();
    macro_rules! t {
        ($f:ident, $max_amount:expr, $max_length:expr) => {
            do_test(|length, amount| $f(r, length, amount), stringify!($f), $max_amount, $max_length);
        }
    }

    t!(cache_hash, 0, 0);
    t!(cache_vec, 100_000, 0);
    t!(cache_sorted_vec, 500_000, 0);
    t!(cache_btree, 0, 0);
    t!(cache_fisher, 0, 0);
    t!(floyd_hash, 0, 0);
    t!(floyd_vec, 500_000, 0);
    t!(floyd_sorted_vec, 500_000, 0);
    t!(floyd_btree, 0, 0);
    t!(inplace, 0, 1_000_000);
    t!(inplace_rev, 0, 1_000_000);
    t!(pitdicker, 0, 0);
}

#[allow(unreachable_code)]
fn verify(res: SampleResult, length: usize, amount: usize) {
    //let mut hash = HashSet::with_capacity(amount);
    let mut sum = 0;
    for val in res {
        sum += val;
        //assert!(val < length);
        //assert!(hash.insert(val));
    }
    //assert_eq!(hash.len(), amount);
    assert!(sum < length * amount);
}

fn nanos(dur: &Duration) -> u64 {
    dur.as_secs() * 1_000_000_000 + dur.subsec_nanos() as u64
}

enum SampleResult {
    Vec(Vec<usize>),
    Hash(HashSet<usize>),
    BTree(BTreeSet<usize>),
}

enum SampleResultIter {
    Vec(std::vec::IntoIter<usize>),
    Hash(std::collections::hash_set::IntoIter<usize>),
    BTree(std::collections::btree_set::IntoIter<usize>),
}

impl IntoIterator for SampleResult {
    type Item = usize;
    type IntoIter = SampleResultIter;
    fn into_iter(self) -> SampleResultIter {
        match self {
            SampleResult::Vec(vec) => SampleResultIter::Vec(vec.into_iter()),
            SampleResult::Hash(hash) => SampleResultIter::Hash(hash.into_iter()),
            SampleResult::BTree(btree) => SampleResultIter::BTree(btree.into_iter()),
        }
    }
}

impl Iterator for SampleResultIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            SampleResultIter::Vec(vec) => vec.next(),
            SampleResultIter::Hash(hash) => hash.next(),
            SampleResultIter::BTree(btree) => btree.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            SampleResultIter::Vec(vec) => vec.size_hint(),
            SampleResultIter::Hash(hash) => hash.size_hint(),
            SampleResultIter::BTree(btree) => btree.size_hint(),
        }
    }
}

impl ExactSizeIterator for SampleResultIter {}

// ===========================================================================

fn floyd_vec<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut indices = Vec::with_capacity(amount as usize);
    for j in length - amount .. length {
        let t = rng.gen_range(0, j + 1);
        if indices.contains(&t) {
            indices.push(j)
        } else {
            indices.push(t)
        };
    }
    SampleResult::Vec(indices)
}

fn floyd_sorted_vec<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut indices = Vec::with_capacity(amount as usize);
    for j in length - amount .. length {
        let t = rng.gen_range(0, j + 1);
        if let Result::Err(index) = indices.binary_search(&t) {
            indices.insert(index, t);
        } else {
            indices.push(j);
        }
    }
    SampleResult::Vec(indices)
}

fn floyd_hash<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut cache = HashSet::with_capacity(amount);
    for j in length - amount .. length {
        let t = rng.gen_range(0, j + 1);
        if !cache.insert(t) {
            cache.insert(j);
        }
    }
    SampleResult::Hash(cache)
}

fn floyd_btree<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut cache = BTreeSet::new();
    for j in length - amount .. length {
        let t = rng.gen_range(0, j + 1);
        if !cache.insert(t) {
            cache.insert(j);
        }
    }
    SampleResult::BTree(cache)
}

fn inplace<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut indices = Vec::with_capacity(length as usize);
    indices.extend(0usize..length);
    for i in 0..amount {
        let j = rng.gen_range(i, length);
        indices.swap(i, j);
    }
    indices.truncate(amount);

    SampleResult::Vec(indices)
}

fn inplace_rev<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut indices = Vec::with_capacity(length as usize);
    indices.extend(0usize..length);

    for i in (amount..length).rev() {
        // invariant: elements with index > i have been locked in place.
        indices.swap(i, rng.gen_range(0, i + 1));
    }
    indices.truncate(amount);

    SampleResult::Vec(indices)
}


fn cache_hash<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut cache = HashSet::with_capacity(amount);
    let distr = Uniform::new(0, length);
    while cache.len() < amount {
        loop {
            if cache.insert(distr.sample(rng)) {
                break;
            }
        }
    }
    
    SampleResult::Hash(cache)
}

fn cache_fisher<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut cache = HashMap::<usize, usize>::with_capacity(amount);
    let mut indices = Vec::with_capacity(amount);
    for i in 0..amount {
        let j: usize = rng.gen_range(i, length);

        // equiv: let tmp = slice[i];
        let tmp = cache.get(&i).map(|x| *x).unwrap_or(i);

        // equiv: x = slice[j]; slice[j] = tmp;
        let x = cache.insert(j, tmp).unwrap_or(j);

        indices.push(x);
    }

    SampleResult::Vec(indices)
}

fn cache_vec<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let distr = Uniform::new(0, length);
    let mut indices = Vec::with_capacity(amount);
    while indices.len() < amount {
        let mut pos = distr.sample(rng);
        while indices.contains(&pos) {
            pos = distr.sample(rng);
        }
        indices.push(pos);
    }
    
    SampleResult::Vec(indices)
}

fn cache_sorted_vec<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let distr = Uniform::new(0, length);
    let mut indices = Vec::with_capacity(amount);
    'outer: while indices.len() < amount {
        loop {
            let picked = distr.sample(rng);
            if let Result::Err(index) = indices.binary_search(&picked) {
                indices.insert(index, picked);
                continue 'outer;
            }
        }
    }
    
    SampleResult::Vec(indices)
}

fn cache_btree<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut cache = BTreeSet::new();
    let distr = Uniform::new(0, length);
    while cache.len() < amount {
        loop {
            if cache.insert(distr.sample(rng)) {
                break;
            }
        }
    }
    
    SampleResult::BTree(cache)
}


// ===========================================================================

fn pitdicker<R>(rng: &mut R, length: usize, amount: usize) -> SampleResult
    where R: Rng + ?Sized
{
    let mut indices = Vec::with_capacity(amount);
    let mut sampler = SequentialRandomSampler::new(amount, length, rng);
    let mut index: usize = 0;

    while sampler.n > 1 {
        index += sampler.calculate_skip(rng) + 1;
        indices.push(index - 1);
    }

    // Optimization: only one more element left to sample.
    // Pick directly, instead of calculating the number of elements
    // to skip.
    index += (sampler.remaining as f64 * sampler.v_prime) as usize;
    indices.push(index);

    SampleResult::Vec(indices)
}


/*
Jeffrey Vitter introduced algorithm **A** and **D** (Edit: and **B** and **C** in-between) to efficiently sample from a known number of elements sequentially, without needing extra memory. ([*Faster Methods for Random Sampling*](http://www.mathcs.emory.edu/~cheung/papers/StreamDB/RandomSampling/1984-Vitter-Faster-random-sampling.pdf), 1984 and [*Efficient Algorithm for Sequential Random Sampling*](http://www.ittc.ku.edu/~jsv/Papers/Vit87.RandomSampling.pdf), 1987)

K. Aiyappan Nair improved upon it with algorithm **E**. (*An Improved Algorithm for Ordered Sequential Random Sampling*, 1990)
*/


// We convert between `f64` and `usize` without caring about possible round-off
// errors. All `usize` values up to 2^52 are exactly representable in `f64`.
// For comparison: on 32-bit we can't have slices > 2^31, and on 64-bit the
// current virtual address space is limited to 2^48 (256 TiB).
#[derive(Debug)]
struct SequentialRandomSampler {
    // remaining number of elements to sample from.
    pub remaining: usize, // Called `N` in the paper.
    // number of elements that should still be sampled.
    pub n: usize,
    // Values cached between runs:
    v_prime: f64,     // FIXME
    threshold: usize, // threshold before switching from method D to method A.
}

// Threshold before switching from method D to method A.
// Typical values of α can be expected in the range 0.05-0.15. The paper
// suggests 1/13, but we because we can make use of the fast Ziggurat method to
// generate exponential values, our method D is relatively fast so α = 1/10
// seems better.
const ALPHA_INV: usize = 10; // (1.0 / α)

impl SequentialRandomSampler {
    fn new<R: Rng + ?Sized>(n: usize, total: usize, rng: &mut R) -> Self {
        let n = ::std::cmp::min(n, total);
        Self {
            remaining: total,
            n: n,
            v_prime: (rng.gen::<f64>().ln() / (n as f64)).exp(),
            threshold: n * ALPHA_INV,
        }
    }

    // FIXME: should this handle n <= 1?
    fn calculate_skip<R: Rng + ?Sized>(&mut self, rng: &mut R) -> usize {
        if self.remaining > self.threshold {
            self.threshold -= ALPHA_INV;
            self.method_d_skip(rng)
        } else {
            self.threshold -= ALPHA_INV;
            self.method_a_skip(rng)
        }
    }

    fn method_a_skip<R: Rng + ?Sized>(&mut self, rng: &mut R) -> usize {
        let mut skip = 0; // Called `S` in the paper.
        let mut remaining_f = self.remaining as f64;
        let n_f = self.n as f64;

        // Step A1
        let v: f64 = rng.gen();

        // Step A2
        // Search sequentially for the smallest integer S satisfying the
        // inequality  V ≤ ((N - n) / n)^(S+1).
        let mut top = remaining_f - n_f;
        let mut quot = top / remaining_f;
        while quot > v {
            skip += 1;
            top -= 1.0;
            remaining_f -= 1.0;
            quot = quot * top / remaining_f;
        }

        // Prepare variables for the next iteration.
        // Note: the paper(s) forgot to subtract `skip`.
        self.remaining -= skip + 1;
        self.n -= 1;
        skip
    }

    fn method_d_skip<R: Rng + ?Sized>(&mut self, rng: &mut R) -> usize {
        let mut skip; // Called `S` in the paper.

        // Cache a couple of variables and expressions we use multiple times.
        let remaining_f = self.remaining as f64;
        let n_f = self.n as f64;
        let ninv = 1.0 / n_f;
        let nmin1inv = 1.0 / (n_f - 1.0);
        let qu1 = self.remaining - self.n + 1;
        let qu1_f = remaining_f - n_f + 1.0;

        loop {
            // Step D2: Generate U and X.
            // "Generate a random variate U that is uniformly distributed
            // between 0 and 1, and a random variate X that has density
            // function or probability function g(x)."
            //
            //        ⎧  n  ⎛      x  ⎞ n-1
            //        ⎪ --- ⎜ 1 - --- ⎟    ,   0 ≤ x ≤ N;
            // g(x) = ⎨  N  ⎝      N  ⎠
            //        ⎪
            //        ⎩  0,                    otherwise;
            //
            // Note: we rename U → u and X → x.
            let mut x;
            loop {
                x = remaining_f * (1.0 - self.v_prime);
                skip = x as usize;
                if skip < qu1 { break; }
                self.v_prime = (-rng.sample(Exp1) * ninv).exp();
            }
            let skip_f = skip as f64;

            // Step D3: Accept?
            // Do a quick approximation to decide whether `x` should be rejected
            //
            // If `x` ≤ h(⌊x⌋)/cg(x), then set `skip` = ⌊x⌋ and go to Step D5.
            // We use the fast method from formula (2.7 + 2.8) here.
            //

            //      ⎛     N U     ⎞ n-1    N - n + 1       N - X
            // V' = ⎜ ----------- ⎟     --------------- * -------
            //      ⎝  N - n + 1  ⎠      N - n - S + 1       N
            //
            // V' ≤ 1?
            //
            // Note: `qu1_f == N - n + 1`
            let y1 = (1.0 / qu1_f - rng.sample(Exp1) * nmin1inv).exp();
            self.v_prime =
                y1 * (qu1_f / (qu1_f - skip_f)) * (1.0 - x / remaining_f);
            if self.v_prime <= 1.0 { break; }

            // Step D4: Accept?
            // Try again using the more expensive method:
            // If U ≤ f(⌊X⌋)/cg(X), then set S := ⌊X⌋.
            // Otherwise, return to Step D2.
            let mut y2 = 1.0;
            let mut top = remaining_f - 1.0;

            let mut bottom;
            let limit;
            if self.n > skip + 1 {
                bottom = remaining_f - n_f;
                limit = self.remaining - skip;
            } else {
                bottom = remaining_f - skip_f - 1.0;
                limit = qu1;
            }
            let mut t = self.remaining - 1;
            while t >= limit {
                y2 = y2 * top / bottom;
                top -= 1.0;
                bottom -= 1.0;
                t -= 1;
            }

            if remaining_f / (remaining_f - x) >= y1 * (y2.ln() * nmin1inv).exp() {
                self.v_prime = (-rng.sample(Exp1) * nmin1inv).exp();
                break; // Accept!
            }

            // We were unlucky, `x` is rejected.
            // Generate a new V' and go back to the beginning.
            self.v_prime = (-rng.sample(Exp1) * ninv).exp();
        }

        // Prepare variables for the next iteration.
        // V' (`self.v_prime`) is already prepared in the loop)
        self.remaining -= skip + 1;
        self.n -= 1;
        skip
    }
}
