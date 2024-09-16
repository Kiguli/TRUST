<script setup>
import H2 from "@/Atoms/H2.vue";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@headlessui/vue";
import UploadDataPanel from "@/Molecules/UploadDataPanel.vue";
import ManualDataPanel from "@/Molecules/ManualDataPanel.vue";

defineProps({
    title: {
        type: String,
        default: "Add Dataset",
    },
    description: {
        type: String,
        default: "Enter or upload your dataset.",
    },
    form: {
        type: Object,
        required: true,
    },
});

const data = defineModel();

const modes = [{ title: "Manual" }, { title: "Upload" }];
</script>

<template>
    <div class="py-2">
        <div class="mb-1.5">
            <H2>
                {{ title }}
            </H2>
            <p class="text-sm text-gray-400">
                {{ description }}
            </p>
        </div>

        <TabGroup>
            <TabList
                class="flex space-x-1 rounded-xl bg-blue-900/20 p-1 dark:bg-gray-900">
                <Tab
                    v-for="mode in modes"
                    :key="mode"
                    v-slot="{ selected }"
                    as="template">
                    <button
                        :class="[
                            selected
                                ? 'bg-white text-gray-600 shadow dark:bg-violet-900/75 dark:text-white'
                                : 'text-gray-500 hover:bg-white/[0.20] dark:hover:bg-gray-800',
                        ]"
                        class="h-10 w-full rounded-lg py-2 text-sm font-medium leading-5 ring-white/60 focus:outline-none focus:ring-2">
                        {{ mode.title }}
                    </button>
                </Tab>
            </TabList>

            <TabPanels class="mt-2">
                <TabPanel v-for="mode in modes" :key="mode">
                    <UploadDataPanel v-if="mode.title === 'Upload'" :form="form" />
                    <ManualDataPanel v-else v-model="data" />
                </TabPanel>
            </TabPanels>
        </TabGroup>
    </div>
</template>

<style scoped></style>
