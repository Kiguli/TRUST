<script setup>
import { ref } from "vue";
import { LinkIcon } from "@heroicons/vue/16/solid/index.js";
import { Head, useForm } from "@inertiajs/vue3";

import ProblemOptions from "@/Organisms/ProblemOptions.vue";
import DatasetInput from "@/Organisms/DatasetInput.vue";
import H2 from "@/Atoms/H2.vue";
import VectorInput from "@/Organisms/VectorInput.vue";
import Pre from "@/Atoms/Pre.vue";
import Section from "@/Atoms/Section.vue";

import route from "~/utilities/route.js";
import UploadDataPanel from "@/Molecules/UploadDataPanel.vue";

defineProps({
    models: Array,
    timings: Array,
    modes: Array,
    result: null,
});

// -- Data
const X0 = ref();
const U0 = ref();
const X1 = ref();

// -- Selections
const model = ref();
const timing = ref();
const mode = ref();

// -- Matrix inputs
const monomials = ref();
const stateSpace = ref();
const initialState = ref();
const unsafeStates = ref();

const form = useForm({
    model,
    timing,
    mode,
    X0,
    X1,
    U0,
    monomials,
    stateSpace,
    initialState,
    unsafeStates,
});

const submit = () => {
    form.post(route("dashboard.index"), {
        preserveState: true,
        preserveScroll: true,
        only: ["result"],
    });
};
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
                v-model="form.model"
                :options="models"
                title="Model" />
            <ProblemOptions
                v-model="form.timing"
                :options="timings"
                title="Class" />
            <ProblemOptions
                v-model="form.mode"
                :options="modes"
                title="Specification" />
            <DatasetInput v-model="form.X0" title="Add X0" />
            <DatasetInput v-model="form.U0" title="Add U0" />
            <DatasetInput v-model="form.X1" title="Add X1" />
        </Section>

        <!-- Manual inputs -->
        <Section class="border-t bg-gray-100/80 sm:border-t-0 lg:border-r">
            <VectorInput
                v-if="form.model !== 'Linear'"
                v-model="form.monomials"
                description="Enter the lower and upper bounds"
                title="Monomials" />
            <VectorInput
                v-model="form.stateSpace"
                description="Enter the lower and upper bounds"
                title="State Space" />
            <VectorInput
                v-model="form.initialState"
                description="Enter the lower and upper bounds"
                title="Initial Set" />
            <VectorInput
                v-model="form.unsafeStates"
                description="Enter the lower and upper bounds"
                title="Unsafe Sets" />
        </Section>

        <!-- Output -->
        <Section
            class="border-t bg-gray-200/60 sm:col-span-full lg:col-span-1 lg:border-t-0">
            <H2>Output</H2>
            <Pre v-if="result" title="Server results">{{ result }}</Pre>
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

        <div
            class="sticky bottom-0 col-span-full flex min-h-20 w-full justify-between gap-x-4 border-t bg-white px-4 py-2.5 sm:px-8 dark:border-none dark:bg-gray-900 dark:shadow-inner">
            <div class="flex flex-col justify-center">
                <h3
                    class="mb-0.5 text-base font-medium text-gray-800 dark:text-gray-200">
                    <a
                        class="flex w-min cursor-pointer items-center gap-x-0.5"
                        href="https://github.com/kiguli/sintrajbc"
                        target="_blank">
                        <span class="inline-block flex-none">SinTra-SB</span>
                        <LinkIcon class="h-4 w-4 flex-none text-gray-500" />
                    </a>
                </h3>
                <p class="line-clamp-2 text-xs text-gray-400">
                    Single Trajectory Data-Driven Control Synthesis for
                    Stability and Barrier Certificates
                </p>
            </div>
            <div class="flex items-center gap-x-2">
                <button
                    :disabled="form.processing"
                    class="order-1 flex h-min rounded-md bg-blue-600/75 px-4 py-2 text-base text-gray-50 outline-none hover:bg-blue-700/75 focus:ring-blue-600 active:bg-blue-800/75 sm:px-5 sm:py-2.5"
                    type="submit">
                    Calculate
                </button>
                <button
                    class="order-0 flex h-min rounded-md px-4 py-2 text-base text-gray-400 hover:ring-2 hover:ring-inset hover:ring-gray-400/75 active:text-gray-200 active:ring-gray-300 sm:px-5 sm:py-2.5">
                    Reset
                </button>
            </div>
        </div>
    </form>
</template>

<style scoped></style>
