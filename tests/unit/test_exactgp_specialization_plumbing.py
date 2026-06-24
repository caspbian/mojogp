from __future__ import annotations

from mojogp import SingleOutputGP, RBF


class _FakeKernelModule:
    pass


def test_exactgp_ensure_compiled_passes_applied_specialization(monkeypatch):
    import mojogp.gp as gp_module
    import mojogp.loader as loader_module

    gp = SingleOutputGP(RBF(), verbose=False)
    gp.dim = 5
    gp._training_method = "matrix_free"
    gp._set_specialization_request(
        {
            "mode": "applied",
            "profile": {
                "specialization_key": "rbf_tm1_probe",
                "family": "jit_codegen",
                "source": "benchmark",
                "schedule_overrides": {
                    "tm": 1,
                    "use_shmem": True,
                    "j_unroll": 1,
                    "ncols": [6, 1],
                    "block_size": 128,
                    "max_registers": 200,
                    "precompute_inv_ls": False,
                },
                "ncols_hint": [6, 1],
                "module_suffix": "rbf_tm1_probe",
            },
        }
    )

    captured: dict[str, object] = {}

    def _fake_load_kernel_module_engine(*args, **kwargs):
        captured["specialization_decision"] = kwargs.get("specialization_decision")
        captured["ncols_hint"] = kwargs.get("ncols_hint")
        return _FakeKernelModule()

    monkeypatch.setattr(loader_module, "load_kernel_module_engine", _fake_load_kernel_module_engine)
    monkeypatch.setattr(loader_module, "load_engine", lambda **kwargs: object())
    monkeypatch.setattr(
        gp_module,
        "revoke_conflicting_provider_leases_by_name",
        lambda module_name, **kwargs: captured.setdefault("lease_module_name", module_name),
    )

    gp._ensure_compiled()

    decision = captured["specialization_decision"]
    assert decision is not None
    assert decision.applied is True
    assert decision.profile.specialization_key == "rbf_tm1_probe"
    assert captured["ncols_hint"] is None
    assert "_rbf_tm1_probe" in str(captured["lease_module_name"])


def test_exactgp_specialization_metadata_attaches_only_when_enabled():
    gp = SingleOutputGP(RBF(), verbose=False)
    info = {"training_route": "matrix_free"}

    assert gp._maybe_attach_specialization_metadata(info) == {"training_route": "matrix_free"}

    gp._set_specialization_request(
        {
            "mode": "shadow",
            "profile": {
                "specialization_key": "shadow_probe",
                "family": "jit_codegen",
                "source": "benchmark",
                "default_equivalent": True,
            },
        }
    )
    gp._specialization_decision = gp._resolve_specialization_decision(RBF(), 5)
    enriched = gp._maybe_attach_specialization_metadata({"training_route": "matrix_free"})

    assert enriched is not None
    assert enriched["specialization_mode"] == "shadow"
    assert enriched["specialization_key"] == "shadow_probe"
