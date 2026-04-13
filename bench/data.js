window.BENCHMARK_DATA = {
  "lastUpdate": 1776075849820,
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
          "id": "598daf2c5e10cdaee3ddd319561255ffa913846a",
          "message": "Update benchmarks; include OpenCLBackend",
          "timestamp": "2026-01-23T13:19:15+01:00",
          "tree_id": "2ea4904d4fb5f799586d02042591494d44c5fea3",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/598daf2c5e10cdaee3ddd319561255ffa913846a"
        },
        "date": 1769171202166,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 501227191.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 474580351,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 440407561.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 484779212,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 323043269,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 383475258,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 271591361,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 374916414.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2224754836,
            "unit": "ns",
            "extra": "gctime=0\nmemory=85648\nallocs=712\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 653675238.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=77680\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2683122846,
            "unit": "ns",
            "extra": "gctime=0\nmemory=90112\nallocs=721\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 693020304,
            "unit": "ns",
            "extra": "gctime=0\nmemory=82160\nallocs=643\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1226839821,
            "unit": "ns",
            "extra": "gctime=0\nmemory=76672\nallocs=652\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 600768526,
            "unit": "ns",
            "extra": "gctime=0\nmemory=78064\nallocs=651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2336967712,
            "unit": "ns",
            "extra": "gctime=0\nmemory=81136\nallocs=661\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 606865229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=82544\nallocs=661\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "e55f1120a75db7435b888590e7c24c640169afc2",
          "message": "Use AcceleratedKernels",
          "timestamp": "2026-01-23T13:37:53+01:00",
          "tree_id": "a9417a7437da1da74ebc8f530f6a1296ff125140",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/e55f1120a75db7435b888590e7c24c640169afc2"
        },
        "date": 1769172287353,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 508104558,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 484218057,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 439754390.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 482176823,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 306174122,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 381253630,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 279783422.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 373011324,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2219611943,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 667143661,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2689714011.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108928\nallocs=1011\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 689133773.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1216261919,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 594841650,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2328608361,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 602525380,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "2b13ffbe6411f6fd562103945703fc863482454d",
          "message": "Docs: update to DocumenterVitepress v0.3",
          "timestamp": "2026-02-03T09:32:46+01:00",
          "tree_id": "3bda8334f1b12cc7c0dadd80a87b49b2431072b0",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/2b13ffbe6411f6fd562103945703fc863482454d"
        },
        "date": 1770108035782,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 508424085.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 482423560,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 440663133.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 476881095,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 310446677,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 375270452,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 273428095,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 376566811,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2234715573,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 660492124.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2711742432,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 703891688,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1225214110,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 587205907,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2309351302,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 607830110,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "1763b98d75d652751441131bbc8bfc20d68c1c42",
          "message": "Spreading: use fine-grained locking for CPU threads\n\nInstead of using a single global lock to allow a thread to write its\nresults, we now use a fine-grained approach which only locks the regions\nof the output arrays which will potentially be modified.\n\nThis is inspired by ducc's approach.",
          "timestamp": "2026-02-17T10:36:02+01:00",
          "tree_id": "1e0a838ca40e9aa5d622b1bceb23ec5c44711608",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/1763b98d75d652751441131bbc8bfc20d68c1c42"
        },
        "date": 1771321757042,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 498965608,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29664\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 472777246,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29136\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 432025420,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29664\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 469698746,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29136\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 296483376,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29520\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 368624298,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29136\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 261842754.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29520\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 366071307,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29136\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2222490255,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 651197031.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2691924743,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 691290224.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1221411649,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 588722858,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2329887808,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 603252227,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "52c0df79d80d65edcb2f6d91364e7433a66d62fb",
          "message": "Revert \"Spreading: use fine-grained locking for CPU threads\"\n\nThis reverts commit 1763b98d75d652751441131bbc8bfc20d68c1c42.",
          "timestamp": "2026-02-17T10:41:29+01:00",
          "tree_id": "3bda8334f1b12cc7c0dadd80a87b49b2431072b0",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/52c0df79d80d65edcb2f6d91364e7433a66d62fb"
        },
        "date": 1771321809909,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 507930128.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 481374310,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 440987685,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 475781536,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 311053258,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 385202732,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 275935881,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 379745016.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2219313817,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 661085126,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2695294699.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 698656388,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1232776061,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 604267424,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2325375135,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 613915379,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "1204a21e6b9fadfad740284f0b7f64f3bbe67caf",
          "message": "Remove benchmarks/LocalPreferences.toml",
          "timestamp": "2026-02-17T13:48:03+01:00",
          "tree_id": "b21a3c3b96883e6466dcbff348158e8a6cf99e6b",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/1204a21e6b9fadfad740284f0b7f64f3bbe67caf"
        },
        "date": 1771332857473,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 508290742.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 480000204,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 440639009,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 476131562,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 315226047.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 368873801,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 286029228.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 392876355,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2232387127,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 649380975,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2677744802,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 682209684,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1205306180,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 588247462,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2317078226,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 594395321,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "40dac9e38c7e561b187acc3cf19eda50d0e28db6",
          "message": "Bump julia-actions/cache from 2 to 3 (#77)\n\nBumps [julia-actions/cache](https://github.com/julia-actions/cache) from 2 to 3.\n- [Release notes](https://github.com/julia-actions/cache/releases)\n- [Commits](https://github.com/julia-actions/cache/compare/v2...v3)\n\n---\nupdated-dependencies:\n- dependency-name: julia-actions/cache\n  dependency-version: '3'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-03-09T09:47:58+01:00",
          "tree_id": "396bba1597c6166b17fe5425ab822a3774cb08cf",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/40dac9e38c7e561b187acc3cf19eda50d0e28db6"
        },
        "date": 1773046637031,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 585426807,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 469634365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 439528929.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 468074486,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 363531870,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 402892282,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 289957485,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 405701872,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 7221023225,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 731817253,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 3334164436,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 859077450.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 3760788466,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 683341064.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 3072544419,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 717311605,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e48510aa3eba315fa7662b22685b12b17472a00c",
          "message": "Bump codecov/codecov-action from 5 to 6 (#78)\n\nBumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 5 to 6.\n- [Release notes](https://github.com/codecov/codecov-action/releases)\n- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/codecov/codecov-action/compare/v5...v6)\n\n---\nupdated-dependencies:\n- dependency-name: codecov/codecov-action\n  dependency-version: '6'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-03-30T11:34:58+02:00",
          "tree_id": "450b334d93a834ab1d9fa14f68cea844fc9b51df",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/e48510aa3eba315fa7662b22685b12b17472a00c"
        },
        "date": 1774863850921,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 528518150,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 506048705,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 455268115,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 496261625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 356652279.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 383017433,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 285052783,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 385926960,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2271860783,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 683430798.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2726831379,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 706928013,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1247299730,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 625416943,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2351670504,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 624187017.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "41898282+github-actions[bot]@users.noreply.github.com",
            "name": "github-actions[bot]",
            "username": "github-actions[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "82cbfaeef0afc845576439ee8990195223685ad8",
          "message": "CompatHelper: bump compat for CUDA in [weakdeps] to 6, (keep existing compat) (#79)\n\nCo-authored-by: CompatHelper Julia <compathelper_noreply@julialang.org>",
          "timestamp": "2026-04-13T11:54:41+02:00",
          "tree_id": "b757ee21cf74fe4f31688c1ccf4f3242aa208847",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/82cbfaeef0afc845576439ee8990195223685ad8"
        },
        "date": 1776074604173,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 536214419,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 492431407,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 470066800,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 495951757,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 334391379,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 395328491,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 298888200,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 397907796,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2250266045,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 667251204,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2666009283,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 707965416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1224279195,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 613755173,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2346841161,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 628842052.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
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
          "id": "a2b6e065862a2a77e9b15575cbeeb7d44cdc74e1",
          "message": "CUDA extension now depends on CUDACore + cuFFT",
          "timestamp": "2026-04-13T12:17:16+02:00",
          "tree_id": "50dcc4b9aafa00b81875622b59cdb6769d818289",
          "url": "https://github.com/jipolanco/NonuniformFFTs.jl/commit/a2b6e065862a2a77e9b15575cbeeb7d44cdc74e1"
        },
        "date": 1776075847080,
        "tool": "julia",
        "benches": [
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 487947229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 462277882,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 423946894,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29712\nallocs=236\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 464300536,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 1",
            "value": 302567359,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/atomics/Type 2",
            "value": 360356656.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 1",
            "value": 266852664,
            "unit": "ns",
            "extra": "gctime=0\nmemory=29568\nallocs=233\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "CPU/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/no atomics/Type 2",
            "value": 367189006,
            "unit": "ns",
            "extra": "gctime=0\nmemory=28992\nallocs=230\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 2219560365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=104352\nallocs=1001\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 628073111.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96384\nallocs=922\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2673881057,
            "unit": "ns",
            "extra": "gctime=0\nmemory=108816\nallocs=1010\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/ComplexF64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 670037750.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=100864\nallocs=932\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 1",
            "value": 1203983083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95376\nallocs=941\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/global_memory/Type 2",
            "value": 567005296,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96768\nallocs=940\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 1",
            "value": 2299546641,
            "unit": "ns",
            "extra": "gctime=0\nmemory=99840\nallocs=950\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "OpenCLBackend/Float64/Ns = (128, 128, 128)/ρ = 1.0/σ = 1.5/m = HalfSupport(4)/shared_memory/Type 2",
            "value": 586398702,
            "unit": "ns",
            "extra": "gctime=0\nmemory=101248\nallocs=950\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      }
    ]
  }
}