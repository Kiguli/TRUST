<script setup>
import H2 from "@/Atoms/H2.vue";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@headlessui/vue";
import UploadDataPanel from "@/Molecules/UploadDataPanel.vue";
import ManualDataPanel from "@/Molecules/ManualDataPanel.vue";

const data = defineModel();

const modes = [{ title: "Manual" }, { title: "Upload" }];
</script>

<template>
    <div class="py-1">
        <H2> Add Dataset </H2>

        <TabGroup>
            <TabList class="flex space-x-1 rounded-xl bg-blue-900/20 p-1">
                <Tab
                    as="template"
                    v-slot="{ selected }"
                    v-for="mode in modes"
                    :key="mode">
                    <button
                        class="h-10 w-full rounded-lg py-2 text-sm font-medium leading-5 ring-white/60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2"
                        :class="[
                            selected
                                ? 'bg-white text-gray-600 shadow'
                                : 'text-gray-500 hover:bg-white/[0.20]',
                        ]">
                        {{ mode.title }}
                    </button>
                </Tab>
            </TabList>

            <TabPanels class="mt-2">
                <TabPanel v-for="mode in modes" :key="mode">
                    <UploadDataPanel v-if="mode.title === 'Upload'" />
                    <ManualDataPanel v-model="data" v-else />
                </TabPanel>
            </TabPanels>
        </TabGroup>
    </div>
</template>

<style scoped></style>
