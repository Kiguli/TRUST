<script setup>
import { computed, onMounted, ref, watchEffect } from "vue";
import { Head, router, useForm } from "@inertiajs/vue3";
import { useInterval, watchDebounced } from "@vueuse/core";

import H2 from "@/Atoms/H2.vue";
import H3 from "@/Atoms/H3.vue";
import CopyPre from "@/Atoms/CopyPre.vue";
import Pre from "@/Atoms/Pre.vue";
import Section from "@/Atoms/Section.vue";
import UploadDataPanel from "@/Molecules/UploadDataPanel.vue";
import DatasetInput from "@/Organisms/DatasetInput.vue";
import ProblemOptions from "@/Organisms/ProblemOptions.vue";
import VectorInput from "@/Organisms/VectorInput.vue";
import Input from "@/Atoms/Input.vue";
import Label from "@/Atoms/Label.vue";
import VectorInputSet from "@/Organisms/VectorInputSet.vue";
import route from "~/utilities/route.js";

import { LinkIcon } from "@heroicons/vue/16/solid/index.js";
import Acrynom from "@/Atoms/Acrynom.vue";
import MatrixInput from "@/Organisms/MatrixInput.vue";
import P from "@/Atoms/P.vue";

const props = defineProps({
    models: Array,
    timings: Array,
    modes: Array,
    monomials: Array | Boolean,
    result: null,
});

// TODO: manually save state with remember and restore
// https://inertiajs.com/remembering-state#manually-saving-state
const data = {
    model: null,
    timing: null,
    mode: null,
    X0: null,
    X1: null,
    U0: null,
    monomials: "",
    theta_x: [],
    stateSpace: [],
    initialState: [],
    unsafeStates: [[]],
};

const form = useForm("TRUST", data);

const samples = ref(0);
const dimension = ref(1);

const deltaTime = ref(0);

const submit = () => {
    form.transform((data) => {
        ["X0", "U0", "X1"].forEach((key) => {
            data[key] = Array.isArray(data[key]) ? JSON.stringify(data[key]) : data[key];
        });

        ["monomials", "theta_x", "stateSpace", "initialState", "unsafeStates"].forEach((key) => {
            data[key] = JSON.stringify(data[key]);
        });

        return data;
    }).post(route("dashboard.index"), {
        forceFormData: true,
        preserveState: true,
        preserveScroll: true,
        only: ["result"],
        onStart: () => {
            deltaTime.value = useInterval(1000);
        },
        onSuccess: () => {
            deltaTime.value = null;
        },
    });
};

const resetForm = () => {
    if (form.isDirty && !confirm("Are you sure you want to reset the form?")) return;

    form.cancel();

    props.result.value = null;

    form.defaults = router.restore("TRUST");
};

const calculateTxt = computed(() => {
    const delta = deltaTime?.value;
    const deltaHtml = delta?.value > 0 ? `<span class="text-sm opacity-70 font-mono">(${delta?.value}s)</span>` : "";

    // Show keyboard shortcut
    const platform = navigator.userAgent.toLowerCase();
    const shortcut = (platform.includes("mac") ? "Cmd" : "Ctrl") + "+Enter";
    const shortcutHtml = `<kbd class="text-sm opacity-70">(${shortcut})</kbd>`;
    const calculateHtml = `Calculate ${shortcutHtml}`;

    return form.processing ? `Calculating... ${deltaHtml}` : calculateHtml;
});

const monomials = ref();

watchEffect(() => {
    dimension.value = Math.max(form.X0?.length ?? 1, 1);
    samples.value = form.X0?.[0]?.length ?? 0;

    // TODO: show datasets error when inconsistent shapes
});

watchDebounced(monomials, () => {
    form.errors.monomials = undefined;

    if (!monomials.value) {
        return;
    }

    const monomialTerms = monomials.value.split(";").map((term) => {
        // error if contains comma
        if (term.includes(",")) {
            form.errors.monomials = "Monomial terms should be split by semicolon";
            return;
        }

        return term.trim();
    });

    if (!monomialTerms) {
        form.errors.monomials = "Invalid monomial terms";
        return;
    }

    router.post(
        route("dashboard.index"),
        {
            monomials: {
                terms: monomialTerms,
                dimensions: dimension.value,
            },
        },
        {
            preserveState: true,
            preserveScroll: true,
            only: ["monomials"],
            onSuccess: () => {
                if (props.monomials) {
                    form.monomials = props.monomials;
                } else {
                    form.errors.monomials = `Monomials must be in terms of x1` +
                        (dimension.value > 1 ? ` to x${dimension.value}` : "");
                }
            },
        },
    );
}, 500);

const submitBtn = ref();

onMounted(() => {
    window.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            submitBtn.value?.click();
        }
    });
});

</script>

<template>
    <Head title="Dashboard" />

    <form
        class="grid sm:grid-cols-2 lg:h-screen lg:grid-cols-3"
        @submit.prevent="submit">
        <!-- Input options -->
        <Section class="bg-gray-50 sm:border-r">
            <div>
                <H2> Upload MOSEK License </H2>
                <p class="text-sm dark:text-gray-100">
                    Please upload your MOSEK license file for this session.
                </p>
                <P>
                    If you do not have a license, academics can get a free trial
                    from the MOSEK website, or a paid version is available.
                </P>

                <UploadDataPanel
                    :form="form"
                    description="Upload or drag and drop your MOSEK license file." />
            </div>

            <ProblemOptions
                v-model="form.timing"
                :options="timings"
                title="Class" />
            <ProblemOptions
                v-model="form.model"
                :options="models"
                title="Model" />
            <ProblemOptions
                v-model="form.mode"
                :options="modes"
                title="Specification" />

            <DatasetInput v-model="form.X0" :form="form" title="Add X0" />
            <DatasetInput v-model="form.U0" :form="form" title="Add U0" />
            <DatasetInput v-model="form.X1" :form="form" title="Add X1" />
        </Section>

        <!-- Manual inputs -->
        <Section class="border-t bg-gray-100/80 sm:border-t-0 lg:border-r">
            <div
                class="py-1">
                <H2>Dimensions</H2>
                <P>
                    The auto-calculated dimensions from your dataset.
                </P>
                <div class="mt-2 flex rounded-md shadow-sm">
                    <Label class="opacity-30" for="dimensions">
                        Dimensions
                    </Label>
                    <Input
                        id="dimensions"
                        :content="dimension"
                        :placeholder="dimension"
                        :value="dimension"
                        disabled
                        type="text" />
                </div>
            </div>

            <div
                v-if="form.model === 'Non-Linear Polynomial'"
                class="py-1">
                <H2>Monomials</H2>
                <P>
                    Enter the monomials in terms of
                    <Pre>x1</Pre>
                    <span v-if="dimension > 1">
                        to
                        <Pre>x{{ dimension }}</Pre>.
                    </span>
                </P>
                <div class="mt-2 flex rounded-md shadow-sm">
                    <Label for="monomials">
                        Monomials
                    </Label>
                    <Input
                        id="monomials"
                        v-model="monomials"
                        :has-error="form.errors.monomials !== undefined"
                        aria-autocomplete="none"
                        autocapitalize="off"
                        placeholder="e.g. x1; 2 * x2; x3 - x1"
                        required
                        type="text" />
                </div>
                <p v-if="form.errors.monomials" class="text-xs mt-2 text-red-600">{{ form.errors.monomials }}</p>
            </div>

            <!-- Allow user to enter matrix Theta(x) -->
            <div v-if="form.timing === 'Discrete-Time' && form.model === 'Non-Linear Polynomial'">
                <MatrixInput
                    v-model="form.theta_x"
                    :dimension="dimension"
                    :monomials="form.monomials"
                >
                    <template #title>
                        <H2>Matrix &theta;(x)</H2>
                    </template>

                    <template #description>
                        <P>
                            Enter the matrix &theta;(x) in terms of the monomials.
                        </P>
                        <P class="text-xs text-gray-400 dark:text-gray-500">
                            The matrix should be
                            <Pre class="px-1 py-0.5 rounded bg-gray-500/10 inline">N x n</Pre>
                            where
                            <Pre class="px-1 py-0.5 rounded bg-gray-500/10 inline">N</Pre>
                            is the number of monomial terms and
                            <Pre class="px-1 py-0.5 rounded dark:bg-gray-500/10 bg-gray-400/10 inline">n</Pre>
                            is the number of dimensions.
                        </P>
                    </template>
                </MatrixInput>
            </div>

            <div v-if="form.mode !== 'Stability'" class="space-y-2">
                <VectorInput
                    v-model="form.stateSpace"
                    :dimensions="dimension"
                    description="Enter the lower and upper bounds."
                    title="State Space" />
                <VectorInput
                    v-model="form.initialState"
                    :dimensions="dimension"
                    description="Enter the lower and upper bounds."
                    title="Initial Set" />
                <VectorInputSet
                    v-model="form.unsafeStates"
                    :dimensions="dimension"
                    description="Enter the lower and upper bounds."
                    title="Unsafe Sets" />
            </div>
        </Section>

        <!-- Output -->
        <Section
            class="border-t bg-gray-200/60 sm:col-span-full lg:col-span-1 lg:border-t-0">
            <H2>Output</H2>

            <div v-if="result && !form.processing">
                <div
                    class="bg-gray-700/50 w-full px-2 py-1 font-mono text-sm space-y-1 text-gray-100 dark:text-gray-200 overflow-clip rounded dark:shadow-md dark:shadow-gray-950/20">
                    <p class="text-sm">
                        <span class="font-bold">INFO:</span>
                        Calculated in
                        <Pre class="inline-block">{{ result.time_taken }}</Pre>
                    </p>
                    <p class="text-sm">
                        <span class="font-bold">INFO:</span>
                        Peak memory usage
                        <Pre class="inline-block">{{ result.memory_used }}</Pre>
                    </p>
                </div>

                <div v-if="result.error"
                     class="mt-1.5 w-full font-mono text-sm text-gray-100 dark:text-gray-200 rounded overflow-clip dark:shadow-md dark:shadow-red-950/20">
                    <p class="bg-red-800 px-2 py-1">
                        <span class="font-bold">Error:</span>
                        {{ result.error }}
                    </p>
                    <p v-if="result.description" class="bg-red-800/50 px-2 py-1 line-clamp-6 overflow-scroll">
                        {{ result.description ?? "" }}
                    </p>
                </div>

                <div v-else>
                    <div class="my-6">
                        <H3>
                            {{ form.mode !== "Stability" ? "Barrier" : "Lyapunov" }}
                        </H3>
                        <CopyPre
                            :title="(form.mode !== 'Stability' ? 'B(x) = ' : 'V(x) = ') + Object.keys(result.function?.expression)[0]">
                            <span v-html="Object.values(result.function?.expression)[0]"></span>
                        </CopyPre>
                        <CopyPre title="P">

                            {{ result.function.values.P }}
                        </CopyPre>
                    </div>

                    <div class="my-6">
                        <H3>Controller</H3>
                        <CopyPre :title="'u = ' + Object.keys(result.controller.expression)[0]">
                            <span v-html="Object.values(result.controller.expression)[0]"></span>
                        </CopyPre>
                        <CopyPre :title="Object.keys(result.controller.values)[0]">
                            {{ Object.values(result.controller.values)[0] }}
                        </CopyPre>
                    </div>

                    <div v-if="form.mode !== 'Stability'" class="my-6">
                        <H3>Level Sets</H3>
                        <CopyPre title="&gamma;">
                            {{ result.gamma }}
                        </CopyPre>
                        <CopyPre title="&lambda;">
                            {{ result.lambda }}
                        </CopyPre>
                    </div>
                </div>
            </div>

            <div v-else class="flex items-center gap-x-2">
                <div class="flex justify-center pt-px">
                    <span class="relative flex h-3 w-3">
                        <span
                            :class="[form.processing ? 'bg-amber-400' : 'bg-blue-400']"
                            class="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"></span>
                        <span
                            :class="[form.processing ? 'bg-amber-500' : 'bg-blue-500']"
                            class="relative inline-flex h-3 w-3 rounded-full"></span>
                    </span>
                </div>

                <P>
                    <span v-if="form.processing">
                        In progress. Please wait...
                    </span>
                    <span v-else>
                        Connected. Waiting for input...
                    </span>
                </P>
            </div>
        </Section>

        <div
            class="sticky bottom-0 col-span-full flex min-h-20 w-full justify-between gap-x-4 border-t bg-white px-4 py-2.5 sm:px-8 dark:border-none dark:bg-gray-900 dark:shadow-inner">
            <div class="flex flex-col justify-center">
                <h3
                    class="mb-0.5 text-base font-medium text-gray-800 dark:text-gray-200">
                    <a
                        class="flex w-min cursor-pointer items-center gap-x-0.5"
                        href="https://github.com/kiguli/sintrajbc"
                        target="_blank">
                        <span class="inline-block flex-none">TRUST</span>
                        <LinkIcon class="h-4 w-4 flex-none text-gray-500 dark:text-gray-400" />
                    </a>
                </h3>
                <span class="line-clamp-2 text-xs text-gray-400">
                    Stabili<Acrynom>T</Acrynom>y and Safety Cont<Acrynom>R</Acrynom>oller Synthesis for Black-Box
                    Systems <Acrynom>U</Acrynom>sing a <Acrynom>S</Acrynom>ingle <Acrynom>T</Acrynom>rajectory
                </span>
            </div>
            <div class="flex items-center gap-x-2">
                <button
                    ref="submitBtn"
                    :disabled="form.processing"
                    class="order-1 flex h-min items-baseline gap-x-1 rounded-md bg-blue-600/75 px-4 py-2 text-base text-gray-50 outline-none ring-2 ring-inset ring-transparent hover:bg-blue-700/75 focus:ring-gray-100 active:bg-blue-800/75 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-blue-600/75 sm:px-5 sm:py-2.5"
                    type="submit"
                    v-html="calculateTxt" />
                <button
                    class="order-0 flex h-min rounded-md px-4 py-2 text-base text-gray-400 hover:ring-2 hover:ring-inset hover:ring-gray-400/75 active:text-gray-200 active:ring-gray-300 sm:px-5 sm:py-2.5"
                    type="reset"
                    @click="resetForm">
                    Reset
                </button>
            </div>
        </div>

    </form>
</template>

<style scoped></style>
