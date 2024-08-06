<script setup>
import {
    RadioGroup,
    RadioGroupDescription,
    RadioGroupLabel,
    RadioGroupOption,
} from "@headlessui/vue";
import { onMounted, ref } from "vue";
import H2 from "@/Atoms/H2.vue";

const props = defineProps({
    title: {
        type: String,
        required: true,
    },
    description: {
        type: String,
        default: "",
    },
    options: {
        type: Array,
        required: true,
    },
});

const model = defineModel({ type: String });

onMounted(() => {
    model.value = props.options[0].title;
});
</script>

<template>
    <div class="py-1">
        <H2>
            {{ title }}
        </H2>

        <RadioGroup v-model="model">
            <RadioGroupLabel class="sr-only">{{ title }}</RadioGroupLabel>
            <div class="grid grid-cols-2 gap-2">
                <RadioGroupOption
                    v-for="option in options"
                    :key="option.title"
                    v-slot="{ active, checked }"
                    :disabled="option.disabled"
                    :value="option.title"
                    as="template">
                    <div
                        :class="[
                            active
                                ? 'ring-2 ring-white/60 ring-offset-2 ring-offset-indigo-300'
                                : '',
                            checked
                                ? 'bg-indigo-900/75'
                                : 'bg-white dark:bg-gray-900',
                            option.disabled
                                ? 'cursor-not-allowed opacity-25'
                                : 'cursor-pointer',
                        ]"
                        class="relative ring-2 ring-offset-2 ring-offset-transparent ring-transparent flex rounded-lg px-4 py-3 shadow-md focus:outline-none">
                        <div class="flex w-full items-center justify-between">
                            <div class="flex items-center">
                                <div class="text-sm">
                                    <RadioGroupLabel
                                        :class="
                                            checked
                                                ? 'text-white'
                                                : 'text-gray-900 dark:text-gray-300'
                                        "
                                        as="p"
                                        class="font-medium line-clamp-1">
                                        {{ option.title }}
                                    </RadioGroupLabel>

                                    <RadioGroupDescription
                                        v-if="option.description !== ''"
                                        :class="
                                            checked
                                                ? 'text-indigo-100'
                                                : 'text-gray-500'
                                        "
                                        as="span"
                                        class="inline line-clamp-1 text-xs">
                                        <span>{{ option.description }}</span>
                                    </RadioGroupDescription>
                                </div>
                            </div>

                            <div v-show="checked" class="shrink-0 text-white">
                                <svg
                                    class="h-5 w-5"
                                    fill="none"
                                    viewBox="0 0 24 24">
                                    <circle
                                        cx="12"
                                        cy="12"
                                        fill="#fff"
                                        fill-opacity="0.2"
                                        r="12" />
                                    <path
                                        d="M7 13l3 3 7-7"
                                        stroke="#fff"
                                        stroke-linecap="round"
                                        stroke-linejoin="round"
                                        stroke-width="1.5" />
                                </svg>
                            </div>
                        </div>
                    </div>
                </RadioGroupOption>
            </div>
        </RadioGroup>
    </div>
</template>

<style scoped></style>
