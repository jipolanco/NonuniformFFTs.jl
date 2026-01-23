window.BENCHMARK_DATA = {
  "lastUpdate": 1769164418755,
  "repoUrl": "https://github.com/jipolanco/NonuniformFFTs.jl",
  "entries": {
    "Julia benchmark result": [
      {
        "commit": {
          "author": {
            "email": "juan-ignacio.polanco@cnrs.fr",
            "name": "Juan Ignacio Polanco",
            "username": "jipolanco"
          },
          "committer": {
            "email": "juan-ignacio.polanco@cnrs.fr",
            "name": "Juan Ignacio Polanco",
            "username": "jipolanco"
          },
          "distinct": true,
          "id": "63f2b1919cb3271f1b7a5c1e3bc538e756ff9818",
          "message": "CI: add continuous benchmark workflow",
          "timestamp": "2026-01-23T10:33:29+01:00",
          "tree_id": "d91d3d1e368a67b601bc2f5136b96a3a2769cbab",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/63f2b1919cb3271f1b7a5c1e3bc538e756ff9818"
        },
        "date": 1769161069835,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/ComplexF64 Type 1",
            "value": 507679256,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38160\nallocs=240\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/Float64 Type 2",
            "value": 376121192,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/Float64 Type 1",
            "value": 305249640,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/ComplexF64 Type 2",
            "value": 478092270,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/ComplexF64 Type 1",
            "value": 451626220,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38160\nallocs=240\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/Float64 Type 2",
            "value": 382336818.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/Float64 Type 1",
            "value": 281525038.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/ComplexF64 Type 2",
            "value": 498439069.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juan-ignacio.polanco@cnrs.fr",
            "name": "Juan Ignacio Polanco",
            "username": "jipolanco"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1c75586bb29aaf1d6902f92036bdc34c8b951a2b",
          "message": "Slightly improve accuracy of KB and BKB kernels (#75)",
          "timestamp": "2026-01-23T11:06:00+01:00",
          "tree_id": "40ba99c23a1e9351b9714f8bb00a78ebb3ac2441",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/1c75586bb29aaf1d6902f92036bdc34c8b951a2b"
        },
        "date": 1769162936118,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/ComplexF64 Type 1",
            "value": 512785410.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38160\nallocs=240\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/Float64 Type 2",
            "value": 384040588,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/Float64 Type 1",
            "value": 316864182,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/ComplexF64 Type 2",
            "value": 480460290,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/ComplexF64 Type 1",
            "value": 444968559.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38160\nallocs=240\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/Float64 Type 2",
            "value": 367278509,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/Float64 Type 1",
            "value": 273660489,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/ComplexF64 Type 2",
            "value": 480352434,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juan-ignacio.polanco@cnrs.fr",
            "name": "Juan Ignacio Polanco",
            "username": "jipolanco"
          },
          "committer": {
            "email": "juan-ignacio.polanco@cnrs.fr",
            "name": "Juan Ignacio Polanco",
            "username": "jipolanco"
          },
          "distinct": true,
          "id": "c9f40346e59ba9f7d2e3d21760944a06e7c621e1",
          "message": "v0.9.1",
          "timestamp": "2026-01-23T11:30:53+01:00",
          "tree_id": "4ca410aadea70f5fbbbe8cb4e689548bd85a1f44",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/c9f40346e59ba9f7d2e3d21760944a06e7c621e1"
        },
        "date": 1769164416900,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/ComplexF64 Type 1",
            "value": 494026863.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38160\nallocs=240\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/Float64 Type 2",
            "value": 365437246,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/Float64 Type 1",
            "value": 300469148,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/atomics/ComplexF64 Type 2",
            "value": 468149037,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/ComplexF64 Type 1",
            "value": 421701531,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38160\nallocs=240\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/Float64 Type 2",
            "value": 367036357.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/Float64 Type 1",
            "value": 268133047,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU: Ns = (128, 128, 128), ρ = 1.0, σ = 1.5, m = HalfSupport(4)/no atomics/ComplexF64 Type 2",
            "value": 467633463,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}