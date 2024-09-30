<script setup>
import { computed, ref, watch, watchEffect } from "vue";
import { Head, router, useForm } from "@inertiajs/vue3";
import { useInterval, watchDebounced } from "@vueuse/core";

import H2 from "@/Atoms/H2.vue";
import H3 from "@/Atoms/H3.vue";
import Pre from "@/Atoms/Pre.vue";
import Section from "@/Atoms/Section.vue";
import UploadDataPanel from "@/Molecules/UploadDataPanel.vue";
import DatasetInput from "@/Organisms/DatasetInput.vue";
import ProblemOptions from "@/Organisms/ProblemOptions.vue";
import StickyFooter from "@/Organisms/StickyFooter.vue";
import VectorInput from "@/Organisms/VectorInput.vue";

import route from "~/utilities/route.js";
import Input from "@/Atoms/Input.vue";
import Label from "@/Atoms/Label.vue";
import VectorInputSet from "@/Organisms/VectorInputSet.vue";

const props = defineProps({
    models: Array,
    timings: Array,
    modes: Array,
    monomials: Array | Boolean,
    result: null,
});

const form = useForm({
    model: null,
    timing: null,
    mode: null,
    X0: null,
    X1: null,
    U0: null,
    monomials: "",
    stateSpace: [],
    initialState: [],
    unsafeStates: [[]],
});

const samples = ref(0);
const dimension = ref(1);

const deltaTime = ref(0);

const submit = () => {
    form.post(route("dashboard.index"), {
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
                console.log(props.monomials);
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
                <p class="text-sm text-gray-400">
                    If you do not have a license, academics can get a free trial
                    from the MOSEK website, or a paid version is available.
                </p>

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
                <p class="text-sm text-gray-400">
                    The auto-calculated dimensions from your dataset.
                </p>
                <div class="mt-2 flex rounded-md shadow-sm">
                    <Label class="opacity-30" for="dimensions">
                        Dimensions
                    </Label>
                    <Input
                        id="dimensions"
                        :value="dimension"
                        disabled
                        type="text" />
                </div>
            </div>

            <div
                v-if="form.model !== 'Linear'"
                class="py-1">
                <H2>Monomials</H2>
                <p class="text-sm text-gray-400">
                    Enter the monomials in terms of
                    <pre class="inline">x1</pre>
                    <span v-if="dimension > 1">
                        to
                        <pre class="inline">x{{ dimension }}</pre>.
                    </span>
                </p>
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
        </Section>

        <!-- Output -->
        <Section
            class="border-t bg-gray-200/60 sm:col-span-full lg:col-span-1 lg:border-t-0">
            <H2>Output</H2>

            <div v-if="result">
                <div>
                    <p class="text-sm text-gray-400">
                        Calculated in
                        <pre class="inline-block">{{ result.time_taken }}</pre>
                    </p>
                </div>

                <div class="my-6">
                    <H3>Barrier</H3>
                    <Pre title="Barrier expression">
                        <span v-html="result.barrier_function.barrier.expression"></span>
                    </Pre>
                    <Pre title="P">
                        {{ result.barrier_function.barrier.values.P }}
                    </Pre>
                </div>

                <div class="my-6">
                    <H3>Controller</H3>
                    <Pre title="Controller expression">
                        <span v-html="result.barrier_function.controller.expression"></span>
                    </Pre>
                    <Pre title="H">
                        {{ result.barrier_function.controller.values.H }}
                    </Pre>
                </div>

                <div class="my-6">
                    <H3>Level Sets</H3>
                    <Pre title="Gamma">
                        {{ result.barrier_function.gamma }}
                    </Pre>
                    <Pre title="Lambda">
                        {{ result.barrier_function.lambda }}
                    </Pre>
                </div>
            </div>

            <div v-else class="flex items-center gap-x-2">
                <div class="flex justify-center pt-px">
                    <span class="relative flex h-3 w-3">
                        <span
                            class="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75"></span>
                        <span
                            class="relative inline-flex h-3 w-3 rounded-full bg-blue-500"></span>
                    </span>
                </div>

                <p class="text-sm text-gray-400">
                    Connected. Waiting for input...
                </p>
            </div>
        </Section>

        <StickyFooter :form="form" :text="calculateTxt" />
    </form>
</template>

<style scoped></style>
