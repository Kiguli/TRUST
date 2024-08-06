<script setup>
import { ref } from "vue";
import { LinkIcon } from "@heroicons/vue/16/solid/index.js";
import { Head, router, useForm } from "@inertiajs/vue3";

import ProblemOptions from "@/Organisms/ProblemOptions.vue";
import DatasetInput from "@/Organisms/DatasetInput.vue";
import H2 from "@/Atoms/H2.vue";
import VectorInput from "@/Organisms/VectorInput.vue";
import Pre from "@/Atoms/Pre.vue";
import Section from "@/Atoms/Section.vue";

import route from "~/utilities/route.js";

defineProps({
    models: Array,
    timings: Array,
    modes: Array,
    result: null,
});

// -- Data
const dataset = ref();

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
    dataset,
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

    <div class="grid sm:grid-cols-2 lg:h-screen lg:grid-cols-3">
        <!-- Input options -->
        <Section class="bg-gray-50 sm:border-r">
            <ProblemOptions
                v-model="form.model"
                :options="models"
                title="Model" />
            <ProblemOptions
                v-model="form.timing"
                :options="timings"
                title="Timing" />
            <ProblemOptions v-model="form.mode" :options="modes" title="Mode" />
            <DatasetInput v-model="form.dataset" />
        </Section>

        <!-- Manual inputs -->
        <Section class="bg-gray-100/80 border-t sm:border-t-0 lg:border-r">
            <VectorInput
                v-if="form.model !== 'Linear'"
                v-model="form.monomials"
                title="Monomials" />
            <VectorInput v-model="form.stateSpace" title="State Space" />
            <VectorInput v-model="form.initialSet" title="Initial Set" />
            <VectorInput v-model="form.unsafeSets" title="Unsafe Set" />
        </Section>

        <!-- Output -->
        <Section
            class="bg-gray-200/60 sm:col-span-full lg:col-span-1 border-t lg:border-t-0">
            <H2>Output</H2>
            <Pre v-if="result" title="Server results">{{ result }}</Pre>
        </Section>

        <div
            class="sticky bottom-0 col-span-full flex w-full justify-between gap-x-4 border-t bg-white px-4 py-2.5 sm:px-8 dark:bg-gray-900 dark:border-none dark:shadow-inner">
            <div class="flex flex-col justify-center">
                <h3 class="mb-0.5 text-base font-medium text-gray-800 dark:text-gray-200">
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
                    @click="submit">
                    Calculate
                </button>
                <button
                    class="order-0 flex h-min rounded-md px-4 py-2 text-base text-gray-400 hover:ring-2 hover:ring-inset hover:ring-gray-400/75 active:ring-gray-300 active:text-gray-200 sm:px-5 sm:py-2.5">
                    Reset
                </button>
            </div>
        </div>
    </div>
</template>

<style scoped></style>
