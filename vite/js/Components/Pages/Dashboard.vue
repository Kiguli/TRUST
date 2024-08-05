<script setup>
import { ref } from "vue";
import { LinkIcon } from "@heroicons/vue/16/solid/index.js";
import { Head } from "@inertiajs/vue3";

import ProblemOptions from "@/Organisms/ProblemOptions.vue";
import DatasetInput from "@/Organisms/DatasetInput.vue";
import H2 from "@/Atoms/H2.vue";
import VectorInput from "@/Organisms/VectorInput.vue";
import Pre from "@/Atoms/Pre.vue";
import Section from "@/Atoms/Section.vue";

defineProps({
    models: Array,
    timings: Array,
    modes: Array,
});

// -- Data
const data = ref();

// -- Selections
const model = ref();
const timing = ref();
const mode = ref();

// -- Matrix inputs
const monomials = ref();
const stateSpace = ref();
const initialSet = ref();
const unsafeSets = ref();
</script>

<template>
    <Head title="Dashboard" />

    <div class="grid sm:grid-cols-2 lg:h-screen lg:grid-cols-3">
        <!-- Input options -->
        <Section class="bg-gray-50">
            <ProblemOptions v-model="model" :options="models" title="Model" />
            <ProblemOptions
                v-model="timing"
                :options="timings"
                title="Timing" />
            <ProblemOptions v-model="mode" :options="modes" title="Mode" />
            <DatasetInput v-model="data" />
        </Section>

        <!-- Manual inputs -->
        <Section class="bg-gray-100/80">
            <VectorInput
                v-if="model !== 'Linear'"
                v-model="monomials"
                title="Monomials" />
            <VectorInput v-model="stateSpace" title="State Space" />
            <VectorInput v-model="initialSet" title="Initial Set" />
            <VectorInput v-model="unsafeSets" title="Unsafe Set" />
        </Section>

        <!-- Output -->
        <Section class="bg-gray-200/60 sm:col-span-full lg:col-span-1">
            <H2>Config</H2>
            <Pre title="Model">{{ model }}</Pre>
            <Pre title="Timing">{{ timing }}</Pre>
            <Pre title="Mode">{{ mode }}</Pre>
            <hr />
            <Pre title="Data">{{ data }}</Pre>
            <hr />
            <Pre title="Monomials">{{ monomials }}</Pre>
            <Pre title="StateSpace">{{ stateSpace }}</Pre>
            <Pre title="InitialSet">{{ initialSet }}</Pre>
            <Pre title="UnsafeSets">{{ unsafeSets }}</Pre>
        </Section>

        <div
            class="sticky bottom-0 col-span-full flex w-full justify-between gap-x-4 border-t bg-white px-4 py-2.5 sm:px-8">
            <div class="flex flex-col justify-center">
                <h3 class="mb-0.5 text-base text-gray-800 font-medium">
                    <a class="flex w-min items-center gap-x-0.5 cursor-pointer" href="https://github.com/kiguli/sintrajbc" target="_blank">
                        <span class="inline-block flex-none">SinTra-SB</span>
                        <LinkIcon class="flex-none h-4 w-4 text-gray-500" />
                    </a>
                </h3>
                <p class="text-xs text-gray-400 line-clamp-2">
                    Single Trajectory Data-Driven Control Synthesis for
                    Stability and Barrier Certificates
                </p>
            </div>
            <div class="flex items-center gap-x-2">
                <button
                    class="order-1 flex h-min rounded-md bg-blue-600/75 px-4 sm:px-5 py-2 sm:py-2.5 text-base text-gray-50 outline-none ring-2 ring-inset hover:bg-blue-700/75 focus:ring-blue-600 active:bg-blue-800/75">
                    Calculate
                </button>
                <button
                    class="order-0 flex h-min rounded-md px-4 sm:px-5 py-2 sm:py-2.5 text-base text-gray-400 ring-2 ring-inset ring-gray-400/75 hover:bg-gray-400 hover:text-white active:bg-gray-500 active:text-white">
                    Reset
                </button>
            </div>
        </div>
    </div>
</template>

<style scoped></style>
