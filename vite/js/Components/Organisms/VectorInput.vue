<script setup>
import { onMounted, ref } from "vue";

import H2 from "@/Atoms/H2.vue";

defineProps({
    title: {
        type: String,
        required: true,
    },
});

const matrixModel = defineModel({ type: Array, default: () => [] });
const matrix = ref([]);

function updateMatrix(i, event) {
    const oldValue = event.target.value;
    let newValue = oldValue;

    // Remove whitespace
    newValue = newValue.replace(" ", "");
    // Split by comma
    newValue = newValue.split(",");
    // Convert to numbers
    newValue = newValue.map((value) => Number(value));
    // Remove empty values
    newValue = newValue.filter((value) => value);

    // Update the matrix
    matrix.value[i - 1] = newValue;

    // Update the parent
    matrixModel.value = matrix.value.filter((value) => value.length);

    // Restore the user input
    event.target.value = oldValue;
}

onMounted(() => {
    matrixModel.value = [];
});
</script>

<template>
    <div class="py-1">
        <H2>{{ title }}</H2>
        <div
            v-for="i in matrixModel.length + 1"
            :key="i"
            class="mt-2 flex rounded-md shadow-sm">
            <label
                :for="`x${i}`"
                class="inline-flex items-center rounded-l-md border border-r-0 border-gray-300 px-3 text-sm text-gray-500 dark:border-2 dark:border-gray-950 dark:text-gray-200">
                x<sub class="mt-1">{{ i }}</sub>
            </label>
            <input
                :id="`x${i}`"
                aria-autocomplete="none"
                autocapitalize="off"
                autocomplete="off"
                class="block h-10 w-full min-w-0 flex-1 rounded-none rounded-r-md border-0 px-2 py-1.5 text-sm leading-6 text-gray-900 outline-none ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-purple-300 dark:border-none dark:bg-gray-950 dark:text-gray-100 dark:outline-none dark:ring-0"
                placeholder="e.g. 1, 2"
                type="text"
                @input="updateMatrix(i, $event)" />
        </div>

        <div class="flex w-full">
            <button
                class="mt-2 rounded-md bg-purple-600/75 px-4 py-2 text-sm text-white hover:bg-purple-700/75 active:bg-purple-800/75"
                @click="dimensions++">
                Add dimension
            </button>

            <button
                class="ml-2 mt-2 rounded-md px-2.5 py-2 text-sm text-gray-400 hover:text-white hover:ring-2 hover:ring-inset hover:ring-gray-400/75 active:border-gray-500 active:text-white"
                @click="clear">
                Clear
            </button>
        </div>
    </div>
</template>

<style scoped></style>
