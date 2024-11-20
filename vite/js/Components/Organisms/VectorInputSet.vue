<script setup>
import VectorInput from "@/Organisms/VectorInput.vue";
import { onMounted, ref, watch } from "vue";
import H2 from "@/Atoms/H2.vue";

const props = defineProps({
    title: String,
    dimensions: Number,
    description: {
        type: String,
        default: "",
    },
});

const unsafeStates = defineModel({ default: [[]] });

const addUnsafeSet = () => {
    unsafeStates.value.push([]);
};

const removeUnsafeSet = (index) => {
    if (unsafeStates.value.length === 1) {
        return;
    }

    unsafeStates.value.splice(index, 1);
};
</script>

<template>
    <div class="space-y-1 py-1">
        <H2>{{ title }}</H2>
        <div v-for="(state, i) in unsafeStates" :key="i" class="relative">
            <button
                v-if="unsafeStates.length > 1"
                @click="removeUnsafeSet(i)"
                class="absolute right-0 top-3.5 text-xs text-gray-400 hover:underline"
                type="button">
                Remove
            </button>
            <VectorInput
                v-model="unsafeStates[i]"
                :description="description"
                :dimensions="dimensions"
                :subtitle="`Unsafe Set ${i + 1}`" />
        </div>
        <div class="flex">
            <button
                class="mt-2 flex h-10 items-center rounded-md bg-gray-600/75 px-4 text-base text-gray-50 outline-none hover:bg-blue-700/75 ring-2 ring-inset ring-transparent focus:ring-blue-600 active:bg-blue-800/75 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-blue-600/75 sm:px-5"
                type="button"
                @click.prevent="addUnsafeSet">
                Add unsafe set
            </button>
        </div>
    </div>
</template>

<style scoped></style>
