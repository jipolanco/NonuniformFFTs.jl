window.BENCHMARK_DATA = {
  "lastUpdate": 1769171203444,
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
      }
    ]
  }
}