<script setup>
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@headlessui/vue"
import ProblemModeOptions from "@/Organisms/ProblemModeOptions.vue"
import ProblemTimeOptions from "@/Organisms/ProblemTimeOptions.vue"

const modes = [
    { title: "Stability", description: "Summary defn of stability" },
    { title: "Barrier", description: "Brief description of a barrier cert" },
]
const classes = [
    { title: "Discrete-Time Linear", description: "Some defn of discrete" },
    { title: "Continuous-Time Linear", description: "And of continuous linear" },
    { title: "Continuous-Time Polynomial", description: "And continuous non-linear polynomial" },
]
</script>

<template>
    <div class="mx-auto w-full max-w-md space-y-12 px-2 py-12 sm:px-0">
        <ProblemModeOptions :modes="modes" />
        <ProblemTimeOptions :classes="classes" />

        <TabGroup v-if="false">
            <TabList class="flex space-x-1 rounded-xl bg-blue-900/20 p-1">
                <Tab
                    as="template"
                    v-slot="{ selected }"
                    v-for="mode in modes"
                    :key="mode">
                    <button
                        class="w-full rounded-lg py-2.5 text-sm font-medium leading-5 ring-white/60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2"
                        :class="[
                            selected
                                ? 'bg-white text-gray-600 shadow'
                                : 'text-gray-400 hover:bg-white/[0.16] hover:text-white',
                        ]">
                        {{ mode.title }}
                    </button>
                </Tab>
            </TabList>

            <TabPanels class="mt-2">
                <TabPanel v-for="mode in modes" :key="mode"></TabPanel>
            </TabPanels>
        </TabGroup>
    </div>
</template>

<style scoped></style>
