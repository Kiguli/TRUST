<script setup>
import ProblemOptions from "@/Organisms/ProblemOptions.vue";
import DatasetInput from "@/Organisms/DatasetInput.vue";
import H2 from "@/Atoms/H2.vue";
import { ref } from "vue";
import VectorInput from "@/Organisms/VectorInput.vue";
import Pre from "@/Atoms/Pre.vue";
import Section from "@/Atoms/Section.vue";

const models = [
    { title: "Linear", description: "" },
    { title: "Polynomial", description: "" },
];

const timings = [
    {
        title: "Discrete-Time",
        description: "",
    },
    {
        title: "Continuous-Time",
        description: "",
    },
];

const modes = [
    { title: "Stability", description: "" },
    { title: "Safety Barrier", description: "" },
    { title: "Reachability Barrier", description: "", disabled: true },
    { title: "Reach and Avoid Barrier", description: "", disabled: true },
];

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
            <H2>Results</H2>
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
    </div>
</template>

<style scoped></style>
