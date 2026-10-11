"""Admitted benchmark dataset identities and their registered task bindings."""

from __future__ import annotations

from dataclasses import dataclass


COMBO_SEEN2 = "combo_seen2"
CONTEXT_OOD = "context_ood"


@dataclass(frozen=True)
class DatasetBinding:
    code: str
    dataset_id: str
    default_task: str
    allowed_tasks: tuple[str, ...]
    task_manifests: dict[str, str]
    execution_ready: bool = True

    @property
    def cache_dir(self) -> str:
        return f"datasets/{self.dataset_id}"


UNSEEN_SINGLE = "unseen_single"
DATASETS = {
    binding.code: binding
    for binding in (
        DatasetBinding(
            code="spinal-ko",
            dataset_id="gse291460_spinal_ko_xenium",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_reaudited.yaml"},
        ),
        DatasetBinding(
            code="gbm-ohsv",
            dataset_id="gse296577_gbm_ohsv_xenium",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_reaudited.yaml"},
        ),
        DatasetBinding(
            code="breast-fus",
            dataset_id="gse296517_breast_fus_cosmx",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_reaudited.yaml"},
        ),
        DatasetBinding(
            code="gut-microbiota",
            dataset_id="gut_microbiota_merfish",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_reaudited.yaml"},
        ),
        DatasetBinding(
            code="cardiac-mi",
            dataset_id="gse280376_cardiac_repair_xenium",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_reaudited.yaml"},
        ),
        DatasetBinding(
            code="kidney-iri",
            dataset_id="gse269719_kidney_iri_xenium",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_reaudited.yaml"},
        ),
        DatasetBinding(
            code="pf-astro",
            dataset_id="perturbfish_astrocytes",
            default_task=UNSEEN_SINGLE,
            allowed_tasks=(UNSEEN_SINGLE,),
            task_manifests={UNSEEN_SINGLE: "benchmark/tasks/intervention_ood_main.yaml"},
        ),
        DatasetBinding(
            code="lung-drug",
            dataset_id="gse325253_lung_drug_xenium",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_lung.yaml"},
        ),
        DatasetBinding(
            code="sickness",
            dataset_id="merfish_sickness",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_expansion.yaml"},
        ),
        DatasetBinding(
            code="dss-colitis",
            dataset_id="dss_colitis_merfish",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_expansion.yaml"},
        ),
        DatasetBinding(
            code="atm-rt",
            dataset_id="gse246584_atm_radiation_xenium",
            default_task=COMBO_SEEN2,
            allowed_tasks=(COMBO_SEEN2,),
            task_manifests={COMBO_SEEN2: "benchmark/tasks/intervention_ood_compositional.yaml"},
        ),
        DatasetBinding(
            code="ddx54-pd1",
            dataset_id="gse285341_ddx54_pd1_xenium",
            default_task=COMBO_SEEN2,
            allowed_tasks=(COMBO_SEEN2,),
            task_manifests={COMBO_SEEN2: "benchmark/tasks/intervention_ood_compositional.yaml"},
        ),
        DatasetBinding(
            code="adipose",
            dataset_id="gse295862_adipose_weightloss_xenium",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_expansion.yaml"},
        ),
        DatasetBinding(
            code="uc-vdz",
            dataset_id="uc_vedolizumab_cosmx",
            default_task=CONTEXT_OOD,
            allowed_tasks=(CONTEXT_OOD,),
            task_manifests={CONTEXT_OOD: "benchmark/tasks/context_ood_biological_unit.yaml"},
        ),
    )
}
